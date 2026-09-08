from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class StrategySignal:
    direction: int
    strength: float
    source: str
    strategy: str


class StrategyMath:
    @staticmethod
    def safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if pd.isna(value):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def has(cls, row: pd.Series, *columns: str) -> bool:
        return all(column in row and pd.notna(row.get(column)) for column in columns)
