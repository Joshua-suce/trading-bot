from dataclasses import dataclass

from src.indicators.compute import compute_all_indicators
from src.signals.invocation import generate_with_context
from src.signals.ta_signal import TASignal, TechnicalSignal


@dataclass
class FinalSignal:
    direction: int
    confidence: float
    ta_source: str
    decision_reason: str = ""
    strategy: str = "trend"


class SignalAggregator:
    def __init__(
        self,
        decision_threshold: float = 0.20,
    ):
        self.ta = TechnicalSignal()
        self.decision_threshold = decision_threshold

    def generate(self, df, higher_trend_bias: int = 0) -> list[FinalSignal]:
        # "adx" only ever exists after compute_all_indicators has run, so its
        # presence means the caller (the live loop) already computed the
        # full indicator set on this frame - recomputing it here would
        # double every EMA/MACD/ADX/RSI/... pass on every closed candle.
        # Callers that pass raw OHLCV (e.g. backtest replay) still get it
        # computed as before.
        df_ind = df if "adx" in df.columns else compute_all_indicators(df)
        ta_signals = generate_with_context(self.ta, df_ind, higher_trend_bias)
        if isinstance(ta_signals, TASignal):
            ta_signals = [ta_signals]

        results: list[FinalSignal] = []
        for ta_signal in ta_signals:
            if ta_signal.direction == 0:
                continue
            direction, confidence, decision_reason = self._apply_mtf_confirmation(
                ta_signal.direction,
                ta_signal.strength,
                "ta signal",
                ta_signal.strategy,
                higher_trend_bias,
            )
            if confidence < self.decision_threshold:
                continue
            results.append(
                FinalSignal(
                    direction=direction,
                    confidence=min(confidence, 1.0),
                    ta_source=ta_signal.source,
                    decision_reason=decision_reason,
                    strategy=ta_signal.strategy,
                )
            )
        return results

    @staticmethod
    def _apply_mtf_confirmation(
        direction: int,
        confidence: float,
        decision_reason: str,
        strategy: str,
        higher_trend_bias: int,
    ) -> tuple[int, float, str]:
        if direction == 0 or higher_trend_bias == 0:
            return direction, confidence, decision_reason

        trend_strategies = {"trend", "breakout", "transition"}
        if strategy in trend_strategies:
            if higher_trend_bias != direction:
                return (
                    0,
                    0.0,
                    f"higher TF bias {higher_trend_bias} opposes "
                    f"{strategy} direction {direction}",
                )
            confidence = min(confidence * 1.15, 1.0)
            decision_reason += " (HTF aligned, +15%)"

        if strategy == "reversal":
            if higher_trend_bias == direction:
                # HTF agrees with the reversal's new direction (e.g. buying a
                # dip inside a larger uptrend) - the best-supported case.
                confidence = min(confidence * 1.10, 1.0)
                decision_reason += " (HTF aligned, reversal confirmed)"
            else:
                # HTF opposes the reversal too - fighting both the local and
                # higher-timeframe trend at once. Weaken, don't reward it.
                confidence *= 0.5
                decision_reason += " (HTF opposes, reversal weakened)"

        return direction, confidence, decision_reason
