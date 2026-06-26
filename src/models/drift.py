from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureDriftReport:
    feature_psi: dict[str, float]
    drifted_features: list[str]
    max_psi: float
    mean_psi: float
    severe: bool


class FeatureDriftMonitor:
    def __init__(
        self,
        *,
        warning_psi: float = 0.20,
        severe_psi: float = 0.30,
        bins: int = 10,
        min_samples: int = 100,
    ) -> None:
        self.warning_psi = warning_psi
        self.severe_psi = severe_psi
        self.bins = bins
        self.min_samples = min_samples

    def compare(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        *,
        feature_names: list[str] | None = None,
    ) -> FeatureDriftReport:
        reference = np.asarray(reference, dtype=float)
        current = np.asarray(current, dtype=float)
        if reference.ndim != 2 or current.ndim != 2:
            raise ValueError("drift inputs must be two-dimensional")
        if reference.shape[1] != current.shape[1]:
            raise ValueError("drift inputs must have matching feature counts")
        if min(len(reference), len(current)) < self.min_samples:
            raise ValueError("insufficient samples for feature drift measurement")
        names = feature_names or [
            f"feature_{index}" for index in range(reference.shape[1])
        ]
        if len(names) != reference.shape[1]:
            raise ValueError("feature name count does not match drift inputs")
        values = {
            name: self._psi(reference[:, index], current[:, index])
            for index, name in enumerate(names)
        }
        scores = list(values.values())
        drifted = [name for name, score in values.items() if score >= self.warning_psi]
        max_psi = max(scores, default=0.0)
        return FeatureDriftReport(
            feature_psi={name: round(score, 6) for name, score in values.items()},
            drifted_features=drifted,
            max_psi=round(max_psi, 6),
            mean_psi=round(float(np.mean(scores)) if scores else 0.0, 6),
            severe=max_psi >= self.severe_psi,
        )

    def _psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        reference = reference[np.isfinite(reference)]
        current = current[np.isfinite(current)]
        if len(reference) == 0 or len(current) == 0:
            return float("inf")
        edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, self.bins + 1)))
        if len(edges) < 2:
            return 0.0 if np.allclose(reference[0], current) else float("inf")
        edges[0] = -np.inf
        edges[-1] = np.inf
        ref_counts, _ = np.histogram(reference, bins=edges)
        cur_counts, _ = np.histogram(current, bins=edges)
        epsilon = 1e-6
        ref_pct = np.maximum(ref_counts / len(reference), epsilon)
        cur_pct = np.maximum(cur_counts / len(current), epsilon)
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
