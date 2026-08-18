"""Reducer for paired prompt, rendering, hint, and tool interventions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .records_paired import PairedArmObservation
from .statistics import ClusterEstimate, cluster_bootstrap_mean


@dataclass(frozen=True, slots=True)
class PairedArmReduction:
    profile: str
    experiment: str
    metric: str
    baseline_arm: str
    treatment_arm: str
    difference: ClusterEstimate
    complete_pairs: int
    dropped_pairs: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "record_type": "paired_arm_reduction",
            "profile": self.profile,
            "experiment": self.experiment,
            "metric": self.metric,
            "baseline_arm": self.baseline_arm,
            "treatment_arm": self.treatment_arm,
            "difference": self.difference.as_dict(),
            "complete_pairs": self.complete_pairs,
            "dropped_pairs": list(self.dropped_pairs),
        }


def reduce_paired_arms(
    rows: Iterable[PairedArmObservation],
    *,
    baseline_arm: str,
    treatment_arm: str,
    confidence: float = 0.95,
    bootstrap_samples: int = 4000,
    seed: int = 0,
) -> list[PairedArmReduction]:
    """Reduce controlled variants strictly within stable pair IDs."""
    if baseline_arm == treatment_arm:
        raise ValueError("baseline and treatment arms must differ")
    grouped: dict[
        tuple[str, str, str], dict[str, list[PairedArmObservation]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(row.profile, row.experiment, row.metric)][row.pair_id].append(row)
    if not grouped:
        raise ValueError("no paired-arm rows")

    reductions: list[PairedArmReduction] = []
    for (profile, experiment, metric), pairs in sorted(grouped.items()):
        by_cluster: dict[str, list[float]] = defaultdict(list)
        dropped: list[str] = []
        complete_pairs = 0
        for pair_id, pair_rows in sorted(pairs.items()):
            arms: dict[str, PairedArmObservation] = {}
            duplicate = False
            for row in pair_rows:
                if row.arm in arms:
                    duplicate = True
                    break
                arms[row.arm] = row
            baseline = arms.get(baseline_arm)
            treatment = arms.get(treatment_arm)
            if (
                duplicate
                or baseline is None
                or treatment is None
                or not baseline.valid
                or not treatment.valid
                or baseline.cluster_id != treatment.cluster_id
            ):
                dropped.append(pair_id)
                continue
            by_cluster[baseline.cluster_id].append(treatment.value - baseline.value)
            complete_pairs += 1
        if not by_cluster:
            raise ValueError(f"no complete valid pairs for {experiment}/{metric}")
        reductions.append(
            PairedArmReduction(
                profile=profile,
                experiment=experiment,
                metric=metric,
                baseline_arm=baseline_arm,
                treatment_arm=treatment_arm,
                difference=cluster_bootstrap_mean(
                    by_cluster,
                    confidence=confidence,
                    samples=bootstrap_samples,
                    seed=seed,
                ),
                complete_pairs=complete_pairs,
                dropped_pairs=tuple(dropped),
            )
        )
    return reductions
