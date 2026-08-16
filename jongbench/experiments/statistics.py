"""Whole-cluster uncertainty utilities for experiment reducers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import fmean, stdev


@dataclass(frozen=True, slots=True)
class ClusterEstimate:
    mean: float
    lower: float | None
    upper: float | None
    standard_error: float | None
    n_clusters: int
    n_observations: int
    confidence: float

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "mean": self.mean,
            "lower": self.lower,
            "upper": self.upper,
            "standard_error": self.standard_error,
            "n_clusters": self.n_clusters,
            "n_observations": self.n_observations,
            "confidence": self.confidence,
        }


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a percentile of no values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = min(max(probability, 0.0), 1.0) * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def cluster_bootstrap_mean(
    values_by_cluster: dict[str, list[float]],
    *,
    confidence: float = 0.95,
    samples: int = 4000,
    seed: int = 0,
) -> ClusterEstimate:
    """Equal-cluster mean and percentile interval from whole-cluster resampling."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not values_by_cluster:
        raise ValueError("no clusters")
    if any(not values for values in values_by_cluster.values()):
        raise ValueError("clusters must not be empty")

    cluster_means = [fmean(values) for values in values_by_cluster.values()]
    point = fmean(cluster_means)
    error = (
        stdev(cluster_means) / math.sqrt(len(cluster_means))
        if len(cluster_means) > 1
        else None
    )
    lower: float | None = None
    upper: float | None = None
    if len(cluster_means) > 1:
        rng = random.Random(seed)
        n = len(cluster_means)
        draws = sorted(
            fmean(cluster_means[rng.randrange(n)] for _ in range(n))
            for _ in range(samples)
        )
        alpha = (1.0 - confidence) / 2.0
        lower = _percentile(draws, alpha)
        upper = _percentile(draws, 1.0 - alpha)
    return ClusterEstimate(
        mean=point,
        lower=lower,
        upper=upper,
        standard_error=error,
        n_clusters=len(cluster_means),
        n_observations=sum(len(values) for values in values_by_cluster.values()),
        confidence=confidence,
    )
