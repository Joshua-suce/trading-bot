import re
from collections import defaultdict
from dataclasses import dataclass

from src.strategies.base import StrategyMath


@dataclass
class SignalGateStats:
    wins: int = 0
    losses: int = 0
    total: int = 0
    directional_bps_sum: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total > 0 else 0.0

    @property
    def avg_directional_bps(self) -> float:
        return self.directional_bps_sum / self.total if self.total > 0 else 0.0


@dataclass
class SignalGateResult:
    passed: bool
    confidence_multiplier: float
    win_rate: float
    total_samples: int
    avg_directional_bps: float
    reason: str


MIN_WIN_RATE: float = 0.45
MIN_SAMPLES: int = 5
MIN_AVG_DIRECTIONAL_BPS: float = 2.0


_SCORE_TOKEN = re.compile(r"^(?:scalp|structure)_score_")


class SignalGate:
    def __init__(
        self,
        min_win_rate: float = MIN_WIN_RATE,
        min_samples: int = MIN_SAMPLES,
        min_avg_directional_bps: float = MIN_AVG_DIRECTIONAL_BPS,
    ):
        self.min_win_rate = min_win_rate
        self.min_samples = min_samples
        self.min_avg_directional_bps = min_avg_directional_bps
        self._stats: dict[str, SignalGateStats] = defaultdict(SignalGateStats)

    def record_outcome(
        self,
        source: str,
        strategy: str,
        timeframe: str,
        won: bool,
        directional_bps: float = 0.0,
    ) -> None:
        key = self._key(source, strategy, timeframe)
        stats = self._stats[key]
        if won:
            stats.wins += 1
        else:
            stats.losses += 1
        stats.total += 1
        stats.directional_bps_sum += directional_bps

    def evaluate(
        self,
        source: str,
        strategy: str,
        timeframe: str,
    ) -> SignalGateResult:
        key = self._key(source, strategy, timeframe)
        stats = self._stats.get(key)
        if stats is None or stats.total < self.min_samples:
            return SignalGateResult(
                passed=True,
                confidence_multiplier=1.0,
                win_rate=stats.win_rate if stats else 0.0,
                total_samples=stats.total if stats else 0,
                avg_directional_bps=stats.avg_directional_bps if stats else 0.0,
                reason="insufficient data",
            )

        penalties = []
        win_rate_failed = False
        edge_failed = False

        if stats.win_rate < self.min_win_rate:
            win_rate_failed = True
            deficit = self.min_win_rate - stats.win_rate
            wr_mult = max(1.0 - deficit * 2.0, 0.50)
            penalties.append(wr_mult)
            reason_part = (
                f"win rate {stats.win_rate:.2%} below "
                f"{self.min_win_rate:.2%} (x{wr_mult:.2f})"
            )
        else:
            wr_mult = 1.0
            reason_part = f"win rate {stats.win_rate:.2%}"

        if stats.avg_directional_bps < self.min_avg_directional_bps:
            edge_failed = True
            bps_deficit = self.min_avg_directional_bps - stats.avg_directional_bps
            bps_mult = max(
                1.0 - (bps_deficit / self.min_avg_directional_bps) * 0.5,
                0.70,
            )
            penalties.append(bps_mult)
            reason_part += (
                f", edge {stats.avg_directional_bps:.2f} bps below "
                f"{self.min_avg_directional_bps:.2f} (x{bps_mult:.2f})"
            )
        else:
            bps_mult = 1.0

        overall_mult = min(penalties) if penalties else 1.0
        # Either metric failing on its own is enough to block: a source can
        # look fine on average edge while still losing most of the time (a
        # few outsized winners masking a bad win rate), and vice versa.
        passed = not (win_rate_failed or edge_failed)

        return SignalGateResult(
            passed=passed,
            confidence_multiplier=overall_mult,
            win_rate=stats.win_rate,
            total_samples=stats.total,
            avg_directional_bps=stats.avg_directional_bps,
            reason=reason_part,
        )

    def load_from_audit(self, records: list[dict]) -> None:
        for record in records:
            source = str(record.get("ta_source", ""))
            strategy = str(record.get("strategy", ""))
            timeframe = str(record.get("timeframe", ""))
            direction_correct = record.get("direction_correct")
            if direction_correct is None:
                continue
            directional_bps = StrategyMath.safe_float(
                record.get("directional_return_bps"), 0.0
            )
            self.record_outcome(
                source,
                strategy,
                timeframe,
                bool(int(direction_correct)),
                directional_bps,
            )

    @property
    def summary(self) -> dict[str, dict[str, float | int]]:
        return {
            key: {
                "wins": s.wins,
                "losses": s.losses,
                "total": s.total,
                "win_rate": round(s.win_rate, 4),
                "avg_directional_bps": round(s.avg_directional_bps, 4),
            }
            for key, s in sorted(self._stats.items())
            if s.total > 0
        }

    @classmethod
    def normalize_source(cls, source: str) -> str:
        tokens = [token for token in str(source or "unknown").split("+") if token]
        normalized = [token for token in tokens if not _SCORE_TOKEN.match(token)]
        return "+".join(normalized) or "unknown"

    @classmethod
    def _key(cls, source: str, strategy: str, timeframe: str) -> str:
        normalized = cls.normalize_source(source)
        return f"{strategy or 'unknown'}:{timeframe or 'unknown'}:{normalized}"

