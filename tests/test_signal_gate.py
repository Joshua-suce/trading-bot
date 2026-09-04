from src.signals.signal_gate import SignalGate


def test_signal_gate_blocks_repeated_negative_source_family():
    gate = SignalGate(min_samples=3, min_win_rate=0.45, min_avg_directional_bps=2.0)
    records = [
        {
            "strategy": "scalp",
            "timeframe": "1m",
            "ta_source": "scalp_vwap_trend_bull+scalp_score_0.82",
            "direction_correct": 0,
            "directional_return_bps": -20.0,
        },
        {
            "strategy": "scalp",
            "timeframe": "1m",
            "ta_source": "scalp_vwap_trend_bull+scalp_score_0.84",
            "direction_correct": 0,
            "directional_return_bps": -10.0,
        },
        {
            "strategy": "scalp",
            "timeframe": "1m",
            "ta_source": "scalp_vwap_trend_bull+scalp_score_0.90",
            "direction_correct": 1,
            "directional_return_bps": 5.0,
        },
    ]

    gate.load_from_audit(records)

    result = gate.evaluate(
        "scalp_vwap_trend_bull+scalp_score_0.96",
        "scalp",
        "1m",
    )

    assert not result.passed
    assert result.total_samples == 3
    assert result.win_rate == 1 / 3
    assert result.avg_directional_bps < 0


def test_signal_gate_allows_insufficient_history():
    gate = SignalGate(min_samples=5)

    result = gate.evaluate("new_source", "trend", "5m")

    assert result.passed
    assert result.reason == "insufficient data"
