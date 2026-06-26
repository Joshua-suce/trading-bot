import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.audit import AuditStore
from src.config import settings
from src.monitoring.execution_quality import execution_quality_summary
from src.monitoring.governance import build_scope_governance
from src.monitoring.health import HealthChecker


def _trades_frame(audit_store: AuditStore) -> pd.DataFrame:
    trades = audit_store.load_trades(500)
    if not trades:
        return pd.DataFrame(
            columns=[
                "symbol",
                "strategy",
                "side",
                "mode",
                "status",
                "entry_price",
                "quantity",
                "exit_price",
                "pnl",
                "updated_at",
            ]
        )
    return pd.DataFrame(trades)


def _events_frame(audit_store: AuditStore) -> pd.DataFrame:
    events = audit_store.load_recent_events(250)
    if not events:
        return pd.DataFrame(
            columns=["ts_utc", "severity", "event_type", "symbol", "mode", "message"]
        )
    df = pd.DataFrame(events)
    keep = ["ts_utc", "severity", "event_type", "symbol", "mode", "message"]
    return df[[column for column in keep if column in df.columns]]


def _signals_frame(audit_store: AuditStore) -> pd.DataFrame:
    observations = audit_store.load_signal_observations(5000)
    if not observations:
        return pd.DataFrame(
            columns=[
                "candle_timestamp",
                "symbol",
                "timeframe",
                "strategy",
                "direction",
                "confidence",
                "decision",
                "execution_status",
                "outcome_status",
                "directional_return_bps",
                "direction_correct",
            ]
        )
    return pd.DataFrame(observations)


def _executions_frame(audit_store: AuditStore) -> pd.DataFrame:
    attempts = audit_store.load_execution_attempts(5000)
    if not attempts:
        return pd.DataFrame(
            columns=[
                "started_at",
                "symbol",
                "timeframe",
                "strategy",
                "phase",
                "side",
                "status",
                "slippage_bps",
                "order_latency_ms",
                "fill_resolution_latency_ms",
                "protection_latency_ms",
                "fill_source",
            ]
        )
    return pd.DataFrame(attempts)


def _equity_curve(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty or "pnl" not in trades_df.columns:
        return pd.DataFrame(columns=["updated_at", "cumulative_pnl"])
    closed = trades_df[trades_df["pnl"].notna()].copy()
    if closed.empty:
        return pd.DataFrame(columns=["updated_at", "cumulative_pnl"])
    closed["updated_at"] = pd.to_datetime(closed["updated_at"], utc=True)
    closed.sort_values("updated_at", inplace=True)
    closed["cumulative_pnl"] = closed["pnl"].astype(float).cumsum()
    return closed[["updated_at", "cumulative_pnl"]]


def _model_frame() -> pd.DataFrame:
    model_dir = Path(settings.model_dir).expanduser()
    if not model_dir.is_absolute():
        model_dir = Path(__file__).resolve().parents[2] / model_dir
    rows: list[dict[str, Any]] = []
    for symbol in settings.symbols_list:
        for timeframe in settings.timeframes_list:
            scope = f"{symbol}:{timeframe}"
            model_path = model_dir / f"xgb_{symbol}_{timeframe}.json"
            metadata_path = model_path.with_name(f"{model_path.stem}.meta.json")
            metadata: dict[str, Any] = {}
            if metadata_path.exists():
                try:
                    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                    metadata = payload.get("metadata") or {}
                except (OSError, ValueError, TypeError):
                    metadata = {}
            rows.append(
                {
                    "scope": scope,
                    "enabled": scope not in settings.disabled_strategy_scopes_set,
                    "model": "ready" if model_path.exists() else "missing",
                    "label_schema": metadata.get("label_schema", "legacy"),
                    "updated_utc": (
                        datetime.fromtimestamp(
                            model_path.stat().st_mtime,
                            tz=timezone.utc,
                        ).isoformat(timespec="seconds")
                        if model_path.exists()
                        else ""
                    ),
                }
            )
    return pd.DataFrame(rows)


def _trade_metrics(trades_df: pd.DataFrame) -> dict[str, float]:
    if trades_df.empty or "status" not in trades_df.columns:
        return {
            "realized_pnl": 0.0,
            "win_rate": 0.0,
            "closed": 0.0,
            "open": 0.0,
        }
    closed = trades_df[trades_df["status"] == "closed"].copy()
    pnl = (
        pd.to_numeric(closed["pnl"], errors="coerce").dropna()
        if "pnl" in closed.columns
        else pd.Series(dtype=float)
    )
    wins = int((pnl > 0).sum())
    return {
        "realized_pnl": float(pnl.sum()),
        "win_rate": (wins / len(pnl) * 100) if len(pnl) else 0.0,
        "closed": float(len(closed)),
        "open": float((trades_df["status"] == "open").sum()),
    }


def _strategy_metrics(trades_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["strategy", "trades", "wins", "win_rate_pct", "net_pnl"]
    if trades_df.empty or "status" not in trades_df.columns:
        return pd.DataFrame(columns=columns)
    closed = trades_df[trades_df["status"] == "closed"].copy()
    if closed.empty:
        return pd.DataFrame(columns=columns)
    closed["strategy"] = closed.get("strategy", "unknown").fillna("unknown")
    closed["pnl"] = pd.to_numeric(closed.get("pnl"), errors="coerce").fillna(0.0)
    closed["win"] = (closed["pnl"] > 0).astype(int)
    result = (
        closed.groupby("strategy", as_index=False)
        .agg(trades=("pnl", "size"), wins=("win", "sum"), net_pnl=("pnl", "sum"))
        .sort_values(["net_pnl", "trades"], ascending=[False, False])
    )
    result["win_rate_pct"] = result["wins"] / result["trades"] * 100
    return result[columns]


def _signal_metrics(signals_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "strategy",
        "decision",
        "observations",
        "resolved",
        "scored",
        "direction_accuracy_pct",
        "avg_directional_return_bps",
        "entries_opened",
    ]
    if signals_df.empty:
        return pd.DataFrame(columns=columns)
    frame = signals_df.copy()
    frame["strategy"] = frame.get("strategy", "unknown").fillna("unknown")
    frame["decision"] = frame.get("decision", "unknown").fillna("unknown")
    frame["direction_correct"] = pd.to_numeric(
        frame.get("direction_correct"), errors="coerce"
    )
    frame["directional_return_bps"] = pd.to_numeric(
        frame.get("directional_return_bps"), errors="coerce"
    )
    frame["resolved_flag"] = (frame.get("outcome_status") == "resolved").astype(int)
    frame["scored_flag"] = frame["direction_correct"].notna().astype(int)
    frame["correct_flag"] = frame["direction_correct"].fillna(0.0)
    frame["opened_flag"] = (frame.get("execution_status") == "opened").astype(int)
    result = (
        frame.groupby(["strategy", "decision"], as_index=False)
        .agg(
            observations=("decision", "size"),
            resolved=("resolved_flag", "sum"),
            scored=("scored_flag", "sum"),
            correct=("correct_flag", "sum"),
            avg_directional_return_bps=("directional_return_bps", "mean"),
            entries_opened=("opened_flag", "sum"),
        )
        .sort_values(["observations", "strategy"], ascending=[False, True])
    )
    result["direction_accuracy_pct"] = (
        result["correct"] / result["scored"].where(result["scored"] > 0) * 100
    )
    return result[columns]


def _governance_frame(
    signals_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = build_scope_governance(
        signals_df.to_dict("records"),
        trades_df.to_dict("records"),
        window_days=settings.performance_governance_window_days,
        min_signals=settings.performance_governance_min_signals,
        min_trades=settings.performance_governance_min_trades,
        promote_min_accuracy=settings.performance_promote_min_accuracy,
        disable_max_accuracy=settings.performance_disable_max_accuracy,
        promote_min_return_bps=settings.performance_promote_min_return_bps,
        disabled_scopes=settings.disabled_strategy_scopes_set,
    )
    return pd.DataFrame(rows)


def _execution_quality_frame(executions_df: pd.DataFrame) -> pd.DataFrame:
    rows = execution_quality_summary(
        executions_df.to_dict("records"),
        window_hours=settings.execution_quality_window_hours,
        min_attempts=settings.execution_quality_min_attempts,
        warning_slippage_bps=settings.execution_quality_warning_slippage_bps,
        critical_failure_rate=settings.execution_quality_critical_failure_rate,
        warning_protection_latency_ms=(
            settings.execution_quality_warning_protection_latency_ms
        ),
    )
    return pd.DataFrame(rows)


def _apply_control_action(
    audit_store: AuditStore,
    action: str,
    reason: str,
) -> str:
    clean_reason = reason.strip() or f"dashboard {action}"
    if action == "pause":
        audit_store.pause_trading(clean_reason)
        message = "New trade entries paused."
    elif action == "resume":
        audit_store.resume_trading(clean_reason)
        message = "Trading resumed, subject to risk controls."
    elif action == "emergency-stop":
        audit_store.activate_emergency_stop(clean_reason)
        message = "Emergency stop activated."
    elif action == "clear-emergency":
        audit_store.clear_emergency_stop(clean_reason)
        message = "Emergency stop cleared."
    else:
        raise ValueError(f"Unsupported dashboard action: {action}")

    audit_store.safe_record_event(
        "dashboard_admin_action",
        message,
        severity="warning",
        mode="dashboard",
        payload={"action": action, "reason": clean_reason},
    )
    return message


def _render_status_header(audit_store: AuditStore, trades_df: pd.DataFrame) -> None:
    health = HealthChecker(audit_store).check()
    metrics = _trade_metrics(trades_df)
    state_class = {
        "ok": "status-ok",
        "degraded": "status-warn",
        "critical": "status-bad",
    }.get(health.status, "status-warn")
    st.markdown(
        f"""
        <div class="status-line">
          <span class="status-dot {state_class}"></span>
          <strong>{health.status.upper()}</strong>
          <span>{health.trading_state_reason}</span>
          <span class="environment">{settings.binance_environment.upper()}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(5)
    columns[0].metric("Trading", "ACTIVE" if health.trading_allowed else "BLOCKED")
    columns[1].metric("Open positions", int(metrics["open"]))
    columns[2].metric("Realized PnL", f"{metrics['realized_pnl']:,.2f} USDT")
    columns[3].metric("Win rate", f"{metrics['win_rate']:.1f}%")
    columns[4].metric("Critical events (24h)", health.critical_events_24h)


def _render_overview(trades_df: pd.DataFrame, events_df: pd.DataFrame) -> None:
    curve = _equity_curve(trades_df)
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Realized performance")
        figure = go.Figure()
        if not curve.empty:
            figure.add_trace(
                go.Scatter(
                    x=curve["updated_at"],
                    y=curve["cumulative_pnl"],
                    mode="lines",
                    name="Cumulative PnL",
                    line={"color": "#2fb67c", "width": 2},
                    fill="tozeroy",
                    fillcolor="rgba(47,182,124,0.10)",
                )
            )
        figure.update_layout(
            height=330,
            margin={"l": 10, "r": 10, "t": 15, "b": 10},
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        figure.update_yaxes(title_text="Cumulative PnL")
        st.plotly_chart(figure, width="stretch")
    with right:
        st.subheader("Latest activity")
        if events_df.empty:
            st.info("No audit events recorded.")
        else:
            latest = events_df.head(8)
            for row in latest.to_dict("records"):
                symbol = f" | {row.get('symbol')}" if row.get("symbol") else ""
                st.markdown(
                    f"**{str(row.get('severity', 'info')).upper()}** "
                    f"{row.get('event_type', '')}{symbol}"
                )
                st.caption(str(row.get("message", "")))
    strategy_metrics = _strategy_metrics(trades_df)
    st.subheader("Strategy performance")
    if strategy_metrics.empty:
        st.info("No closed strategy-attributed trades recorded yet.")
    else:
        st.dataframe(strategy_metrics, width="stretch", hide_index=True)


def _render_trades(trades_df: pd.DataFrame) -> None:
    symbols = ["All"]
    if not trades_df.empty and "symbol" in trades_df.columns:
        symbols.extend(sorted(trades_df["symbol"].dropna().unique().tolist()))
    filter_columns = st.columns([1, 1, 2])
    symbol = filter_columns[0].selectbox("Symbol", symbols)
    status = filter_columns[1].selectbox(
        "Status",
        ["All", "open", "closed", "reconciliation_pending", "failed"],
    )
    filtered = trades_df.copy()
    if symbol != "All" and not filtered.empty:
        filtered = filtered[filtered["symbol"] == symbol]
    if status != "All" and not filtered.empty:
        filtered = filtered[filtered["status"] == status]

    st.dataframe(filtered, width="stretch", hide_index=True, height=430)
    st.download_button(
        "Download trades CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="trading_history.csv",
        mime="text/csv",
        disabled=filtered.empty,
        width="content",
    )


def _render_audit(events_df: pd.DataFrame) -> None:
    severity_options = ["All"]
    if not events_df.empty and "severity" in events_df.columns:
        severity_options.extend(
            sorted(events_df["severity"].dropna().unique().tolist())
        )
    severity = st.selectbox("Severity", severity_options)
    filtered = events_df
    if severity != "All" and not events_df.empty:
        filtered = events_df[events_df["severity"] == severity]
    st.dataframe(filtered, width="stretch", hide_index=True, height=460)
    st.download_button(
        "Download audit CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="audit_events.csv",
        mime="text/csv",
        disabled=filtered.empty,
    )


def _render_signals(
    signals_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> None:
    st.subheader("Signal evidence")
    st.caption(
        "Forward outcomes measure signal direction independently of order execution."
    )
    governance = _governance_frame(signals_df, trades_df)
    st.markdown("#### Performance governance")
    if governance.empty:
        st.info("No strategy scopes have enough recorded activity for review.")
    else:
        st.dataframe(governance, width="stretch", hide_index=True)
        st.download_button(
            "Download governance CSV",
            data=governance.to_csv(index=False).encode("utf-8"),
            file_name="performance_governance.csv",
            mime="text/csv",
        )
    st.caption(
        "Recommendations are advisory. Apply scope disables through reviewed "
        "configuration changes."
    )

    st.markdown("#### Decision diagnostics")
    metrics = _signal_metrics(signals_df)
    if metrics.empty:
        st.info("No signal observations recorded yet.")
    else:
        st.dataframe(metrics, width="stretch", hide_index=True)

    filtered = signals_df.copy()
    filters = st.columns(3)
    symbol_options = ["All"]
    strategy_options = ["All"]
    decision_options = ["All"]
    if not filtered.empty:
        symbol_options.extend(sorted(filtered["symbol"].dropna().unique().tolist()))
        strategy_options.extend(sorted(filtered["strategy"].dropna().unique().tolist()))
        decision_options.extend(sorted(filtered["decision"].dropna().unique().tolist()))
    symbol = filters[0].selectbox("Signal symbol", symbol_options)
    strategy = filters[1].selectbox("Signal strategy", strategy_options)
    decision = filters[2].selectbox("Signal decision", decision_options)
    if symbol != "All":
        filtered = filtered[filtered["symbol"] == symbol]
    if strategy != "All":
        filtered = filtered[filtered["strategy"] == strategy]
    if decision != "All":
        filtered = filtered[filtered["decision"] == decision]

    visible = [
        "candle_timestamp",
        "symbol",
        "timeframe",
        "strategy",
        "direction",
        "confidence",
        "minimum_confidence",
        "decision",
        "reason",
        "quality_score",
        "execution_status",
        "outcome_status",
        "directional_return_bps",
        "direction_correct",
        "outcome_horizon_seconds",
    ]
    st.dataframe(
        filtered[[column for column in visible if column in filtered.columns]],
        width="stretch",
        hide_index=True,
        height=430,
    )
    st.download_button(
        "Download signals CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="signal_observations.csv",
        mime="text/csv",
        disabled=filtered.empty,
    )


def _render_execution(executions_df: pd.DataFrame) -> None:
    st.subheader("Execution quality")
    quality = _execution_quality_frame(executions_df)
    if quality.empty:
        st.info("No instrumented execution attempts recorded yet.")
    else:
        st.dataframe(quality, width="stretch", hide_index=True)

    visible = [
        "started_at",
        "symbol",
        "timeframe",
        "strategy",
        "phase",
        "side",
        "status",
        "expected_price",
        "actual_price",
        "quantity",
        "slippage_bps",
        "order_latency_ms",
        "fill_resolution_latency_ms",
        "protection_latency_ms",
        "fill_source",
        "recovered_order",
        "reason",
        "correlation_id",
    ]
    st.dataframe(
        executions_df[
            [column for column in visible if column in executions_df.columns]
        ],
        width="stretch",
        hide_index=True,
        height=430,
    )
    st.download_button(
        "Download execution CSV",
        data=executions_df.to_csv(index=False).encode("utf-8"),
        file_name="execution_quality.csv",
        mime="text/csv",
        disabled=executions_df.empty,
    )


def _render_controls(audit_store: AuditStore) -> None:
    allowed, reason = audit_store.trading_allowed()
    controls = audit_store.get_controls()
    st.subheader("Bot controls")
    st.caption(f"Current state: {'active' if allowed else 'blocked'} | {reason}")

    action_columns = st.columns(3)
    with action_columns[0]:
        with st.form("pause_form"):
            pause_reason = st.text_input("Pause reason", key="pause_reason")
            pause_clicked = st.form_submit_button(
                "Pause entries",
                width="stretch",
                disabled=not allowed,
            )
        if pause_clicked:
            st.success(_apply_control_action(audit_store, "pause", pause_reason))
            st.rerun()

    with action_columns[1]:
        with st.form("resume_form"):
            resume_reason = st.text_input("Resume reason", key="resume_reason")
            resume_clicked = st.form_submit_button(
                "Resume entries",
                width="stretch",
            )
        if resume_clicked:
            st.success(_apply_control_action(audit_store, "resume", resume_reason))
            st.rerun()

    with action_columns[2]:
        with st.form("emergency_form"):
            emergency_reason = st.text_input(
                "Emergency reason",
                key="emergency_reason",
            )
            emergency_confirmed = st.checkbox(
                "I confirm the emergency stop",
                key="emergency_confirmed",
            )
            emergency_clicked = st.form_submit_button(
                "Emergency stop",
                width="stretch",
                disabled=not emergency_confirmed,
            )
        if emergency_clicked:
            st.error(
                _apply_control_action(
                    audit_store,
                    "emergency-stop",
                    emergency_reason,
                )
            )
            st.rerun()

    st.divider()
    with st.expander("Clear emergency state"):
        st.warning(
            "Clear this only after positions, protection orders, and exchange "
            "reconciliation have been verified."
        )
        confirmation = st.text_input(
            "Type CLEAR to confirm",
            key="clear_confirmation",
        )
        clear_reason = st.text_input(
            "Clear reason",
            key="clear_reason",
        )
        if st.button(
            "Clear emergency stop",
            disabled=confirmation != "CLEAR",
            type="primary",
        ):
            st.success(
                _apply_control_action(
                    audit_store,
                    "clear-emergency",
                    clear_reason,
                )
            )
            st.rerun()

    st.subheader("Control registry")
    controls_frame = pd.DataFrame(
        [
            {
                "control": key,
                "value": value["value"],
                "reason": value["reason"],
                "updated_at": value["updated_at"],
            }
            for key, value in controls.items()
        ]
    )
    st.dataframe(controls_frame, width="stretch", hide_index=True)


def _render_manual_trading(audit_store: AuditStore) -> None:
    st.subheader("Manual trade execution")
    st.warning(
        "Manual orders bypass strategy signals, but still use account risk limits, "
        "position sizing, reduce-only exits, and protective orders."
    )
    open_trades = audit_store.load_open_trades("trade")
    open_column, close_column = st.columns(2)

    with open_column:
        st.markdown("**Place protected trade**")
        with st.form("manual_open_form"):
            symbol = st.selectbox(
                "Trade symbol",
                settings.symbols_list,
                key="manual_open_symbol",
            )
            enabled_timeframes = [
                timeframe
                for timeframe in settings.timeframes_list
                if f"{symbol}:{timeframe}" not in settings.disabled_strategy_scopes_set
            ]
            timeframe = st.selectbox(
                "Trade timeframe",
                enabled_timeframes,
                key="manual_open_timeframe",
            )
            side = st.segmented_control(
                "Trade side",
                ["long", "short"],
                default="long",
                key="manual_open_side",
            )
            st.caption(
                f"Exchange leverage policy: {settings.max_leverage}x. "
                "Quantity remains stop-risk based."
            )
            require_quality = st.toggle(
                "Require strategy quality gate",
                value=True,
                key="manual_open_quality",
            )
            reason = st.text_input(
                "Entry reason",
                key="manual_open_reason",
            )
            confirmation = st.text_input(
                "Type PLACE to confirm",
                key="manual_open_confirmation",
            )
            submitted = st.form_submit_button(
                "Place trade",
                width="stretch",
                disabled=confirmation != "PLACE",
            )
        if submitted:
            request_id = audit_store.create_manual_trade_request(
                action="open",
                symbol=symbol,
                side=str(side),
                timeframe=timeframe,
                reason=reason or "dashboard protected manual entry",
                options={
                    "require_quality": require_quality,
                },
            )
            st.success(f"Trade request queued: {request_id}")
            st.rerun()

    with close_column:
        st.markdown("**Close selected trade**")
        if not open_trades:
            st.info("No audited open trades are available.")
        else:
            labels = {
                (
                    f"{trade['symbol']} {str(trade['side']).upper()} "
                    f"{trade.get('timeframe') or '-'} | "
                    f"{trade['correlation_id']}"
                ): trade
                for trade in open_trades
            }
            with st.form("manual_close_form"):
                selected = st.selectbox(
                    "Open trade",
                    list(labels),
                    key="manual_close_trade",
                )
                close_reason = st.text_input(
                    "Exit reason",
                    key="manual_close_reason",
                )
                close_confirmation = st.text_input(
                    "Type CLOSE to confirm",
                    key="manual_close_confirmation",
                )
                close_submitted = st.form_submit_button(
                    "Close trade",
                    width="stretch",
                    disabled=close_confirmation != "CLOSE",
                )
            if close_submitted:
                trade = labels[selected]
                request_id = audit_store.create_manual_trade_request(
                    action="close",
                    symbol=str(trade["symbol"]),
                    correlation_id=str(trade["correlation_id"]),
                    reason=close_reason or "dashboard manual close",
                )
                st.success(f"Close request queued: {request_id}")
                st.rerun()

    st.divider()
    st.markdown("**Portfolio exit controls**")
    bulk_left, bulk_right = st.columns(2)
    symbols_with_positions = sorted({str(trade["symbol"]) for trade in open_trades})
    with bulk_left:
        with st.form("close_symbol_form"):
            close_symbol = st.selectbox(
                "Position symbol",
                symbols_with_positions or ["No open symbols"],
                key="bulk_close_symbol",
            )
            symbol_confirm = st.text_input(
                "Type the symbol to confirm",
                key="bulk_close_symbol_confirm",
            )
            symbol_submitted = st.form_submit_button(
                "Close all trades for symbol",
                width="stretch",
                disabled=(not symbols_with_positions or symbol_confirm != close_symbol),
            )
        if symbol_submitted:
            request_id = audit_store.create_manual_trade_request(
                action="close-symbol",
                symbol=close_symbol,
                reason="dashboard symbol-wide close",
            )
            st.success(f"Symbol close queued: {request_id}")
            st.rerun()
    with bulk_right:
        with st.form("close_all_form"):
            close_all_confirm = st.text_input(
                "Type CLOSE ALL to confirm",
                key="bulk_close_all_confirm",
            )
            close_all_submitted = st.form_submit_button(
                "Close every open trade",
                width="stretch",
                disabled=(not open_trades or close_all_confirm != "CLOSE ALL"),
            )
        if close_all_submitted:
            request_id = audit_store.create_manual_trade_request(
                action="close-all",
                symbol="ALL",
                reason="dashboard portfolio-wide close",
            )
            st.error(f"Portfolio close queued: {request_id}")
            st.rerun()

    raw_requests = audit_store.load_manual_trade_requests(50)
    requests = pd.DataFrame(raw_requests)
    st.markdown("**Request history**")
    if requests.empty:
        st.info("No dashboard trade requests recorded.")
    else:
        visible = [
            "created_at",
            "action",
            "symbol",
            "side",
            "timeframe",
            "status",
            "reason",
            "result",
            "completed_at",
        ]
        st.dataframe(
            requests[[column for column in visible if column in requests.columns]],
            width="stretch",
            hide_index=True,
            height=330,
        )
        pending = [
            request for request in raw_requests if request["status"] == "pending"
        ]
        if pending:
            with st.form("cancel_request_form"):
                pending_labels = {
                    (
                        f"{request['action']} {request['symbol']} | "
                        f"{request['request_id']}"
                    ): request
                    for request in pending
                }
                selected_request = st.selectbox(
                    "Pending request",
                    list(pending_labels),
                    key="pending_request",
                )
                cancel_reason = st.text_input(
                    "Cancellation reason",
                    key="cancel_request_reason",
                )
                cancel_submitted = st.form_submit_button(
                    "Cancel pending request",
                    width="content",
                )
            if cancel_submitted:
                request = pending_labels[selected_request]
                cancelled = audit_store.cancel_manual_trade_request(
                    str(request["request_id"]),
                    cancel_reason,
                )
                if cancelled:
                    st.success("Pending request cancelled.")
                else:
                    st.warning("Request was already claimed by the trading process.")
                st.rerun()


def _install_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1500px; padding-top: 1.25rem;}
        h1, h2, h3 {letter-spacing: 0;}
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 6px;
            padding: 14px 16px;
            min-height: 104px;
        }
        .status-line {
            display: flex;
            align-items: center;
            gap: 10px;
            min-height: 42px;
            border-bottom: 1px solid rgba(128, 128, 128, 0.25);
            margin-bottom: 16px;
        }
        .status-dot {width: 9px; height: 9px; border-radius: 50%;}
        .status-ok {background: #2fb67c;}
        .status-warn {background: #e0a82e;}
        .status-bad {background: #d95757;}
        .environment {
            margin-left: auto;
            font-size: 0.78rem;
            font-weight: 700;
            border: 1px solid rgba(128, 128, 128, 0.35);
            border-radius: 4px;
            padding: 3px 7px;
        }
        div[data-testid="stForm"] {border-radius: 6px;}
        @media (max-width: 760px) {
            .block-container {padding-left: 1rem; padding-right: 1rem;}
            .status-line {align-items: flex-start; flex-wrap: wrap;}
            .environment {margin-left: 0;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(audit_store: AuditStore | None = None) -> None:
    st.set_page_config(
        page_title="Trading Operations",
        page_icon=":material/candlestick_chart:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _install_styles()
    audit_store = audit_store or AuditStore()
    trades_df = _trades_frame(audit_store)
    events_df = _events_frame(audit_store)
    signals_df = _signals_frame(audit_store)
    executions_df = _executions_frame(audit_store)

    title_column, refresh_column = st.columns([8, 1])
    with title_column:
        st.title("Trading Operations")
        st.caption(
            f"Binance Futures {settings.binance_environment.upper()} | "
            f"Audit store: {audit_store.db_path}"
        )
    with refresh_column:
        st.write("")
        st.write("")
        if st.button(
            "Refresh",
            width="stretch",
            help="Reload dashboard data",
        ):
            st.rerun()

    _render_status_header(audit_store, trades_df)
    (
        overview_tab,
        trades_tab,
        signals_tab,
        execution_tab,
        manual_tab,
        audit_tab,
        models_tab,
        controls_tab,
    ) = st.tabs(
        [
            "Overview",
            "Trades",
            "Signals",
            "Execution",
            "Manual Trading",
            "Audit",
            "Models",
            "Controls",
        ]
    )
    with overview_tab:
        _render_overview(trades_df, events_df)
    with trades_tab:
        _render_trades(trades_df)
    with signals_tab:
        _render_signals(signals_df, trades_df)
    with execution_tab:
        _render_execution(executions_df)
    with manual_tab:
        _render_manual_trading(audit_store)
    with audit_tab:
        _render_audit(events_df)
    with models_tab:
        st.subheader("ML model inventory")
        st.dataframe(
            _model_frame(),
            width="stretch",
            hide_index=True,
            height=500,
        )
    with controls_tab:
        _render_controls(audit_store)


if __name__ == "__main__":
    render_dashboard()
