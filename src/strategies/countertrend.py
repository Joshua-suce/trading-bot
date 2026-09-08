from __future__ import annotations


class CountertrendStrategy:
    name = "countertrend"

    @staticmethod
    def claims_signal(direction: int, trend_regime: int) -> bool:
        return trend_regime != 0 and direction == -trend_regime

    @staticmethod
    def confidence_multiplier() -> float:
        return 0.65
