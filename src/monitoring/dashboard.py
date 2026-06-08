import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.audit import AuditStore
from src.monitoring.health import HealthChecker


def _trades_frame(audit_store: AuditStore) -> pd.DataFrame:
    trades = audit_store.load_trades(200)
    if not trades:
        return pd.DataFrame(
            columns=[
                "symbol",
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
    events = audit_store.load_recent_events(100)
    if not events:
        return pd.DataFrame(
            columns=["ts_utc", "severity", "event_type", "symbol", "mode", "message"]
        )
    df = pd.DataFrame(events)
    keep = ["ts_utc", "severity", "event_type", "symbol", "mode", "message"]
    return df[[col for col in keep if col in df.columns]]


def _equity_curve(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty or "pnl" not in trades_df.columns:
        return pd.DataFrame(columns=["updated_at", "cumulative_pnl"])
    closed = trades_df[trades_df["pnl"].notna()].copy()
    if closed.empty:
        return pd.DataFrame(columns=["updated_at", "cumulative_pnl"])
    closed["updated_at"] = pd.to_datetime(closed["updated_at"])
    closed.sort_values("updated_at", inplace=True)
    closed["cumulative_pnl"] = closed["pnl"].astype(float).cumsum()
    return closed[["updated_at", "cumulative_pnl"]]


def render_dashboard(audit_store: AuditStore | None = None) -> None:
    audit_store = audit_store or AuditStore()
    health = HealthChecker(audit_store).check()
    controls = audit_store.get_controls()
    trades_df = _trades_frame(audit_store)
    events_df = _events_frame(audit_store)
    open_trades = (
        trades_df[trades_df["status"] == "open"] if not trades_df.empty else trades_df
    )
    closed_trades = (
        trades_df[trades_df["status"] == "closed"] if not trades_df.empty else trades_df
    )

    st.set_page_config(page_title="AI Trading Bot", layout="wide")
    st.title("AI Trading Bot Dashboard")
    st.caption(f"Trade recorder: {audit_store.db_path}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Health", health.status.upper())
    with col2:
        st.metric("Trading", "ON" if health.trading_allowed else "OFF")
    with col3:
        st.metric("Open Trades", str(len(open_trades)))
    with col4:
        realized_pnl = (
            float(closed_trades["pnl"].dropna().astype(float).sum())
            if not closed_trades.empty and "pnl" in closed_trades.columns
            else 0.0
        )
        st.metric("Realized PnL", f"{realized_pnl:.2f}")

    st.subheader("Controls")
    controls_df = pd.DataFrame(
        [
            {
                "key": key,
                "value": value["value"],
                "reason": value["reason"],
                "updated_at": value["updated_at"],
            }
            for key, value in controls.items()
        ]
    )
    st.dataframe(controls_df, use_container_width=True, hide_index=True)

    curve = _equity_curve(trades_df)
    st.subheader("Realized PnL Curve")
    fig = go.Figure()
    if not curve.empty:
        fig.add_trace(
            go.Scatter(
                x=curve["updated_at"],
                y=curve["cumulative_pnl"],
                mode="lines+markers",
                name="Realized PnL",
            )
        )
    fig.update_layout(height=360, hovermode="x unified")
    fig.update_yaxes(title_text="Cumulative PnL")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Open Trades")
    st.dataframe(open_trades, use_container_width=True, hide_index=True)

    st.subheader("Recent Trades")
    st.dataframe(trades_df, use_container_width=True, hide_index=True)

    st.subheader("Recent Audit Events")
    st.dataframe(events_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render_dashboard()
