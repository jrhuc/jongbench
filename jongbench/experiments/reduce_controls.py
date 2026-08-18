"""Reducer for same-wall, same-chair all-control replacements."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean
from typing import Iterable

from .identity import ControlPolicyIdentity
from .records_control import MatchedControlResult
from .statistics import ClusterEstimate, cluster_bootstrap_mean


@dataclass(frozen=True, slots=True)
class MatchedControlReduction:
    profile: str
    control: ControlPolicyIdentity
    model: str
    score_delta: ClusterEstimate
    block_means: tuple[tuple[str, float], ...]
    incomplete_walls: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "record_type": "matched_control_reduction",
            "profile": self.profile,
            "control": self.control.as_dict(),
            "model": self.model,
            "score_delta": self.score_delta.as_dict(),
            "block_means": [
                {"wall_id": wall_id, "mean": mean}
                for wall_id, mean in self.block_means
            ],
            "incomplete_walls": list(self.incomplete_walls),
        }


def reduce_matched_controls(
    rows: Iterable[MatchedControlResult],
    *,
    require_complete_walls: bool = True,
    confidence: float = 0.95,
    bootstrap_samples: int = 4000,
    seed: int = 0,
) -> list[MatchedControlReduction]:
    """Reduce four-chair policy-replacement deltas, one value per wall."""
    grouped: dict[
        tuple[str, ControlPolicyIdentity, str], list[MatchedControlResult]
    ] = defaultdict(list)
    for row in rows:
        grouped[(row.profile, row.control, row.model)].append(row)
    if not grouped:
        raise ValueError("no matched-control rows")

    reductions: list[MatchedControlReduction] = []
    for (profile, control, model), group in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1].policy_id, item[0][2]),
    ):
        by_wall: dict[str, dict[int, MatchedControlResult]] = defaultdict(dict)
        baseline_by_wall: dict[str, str] = {}
        for row in group:
            previous_baseline = baseline_by_wall.setdefault(
                row.wall_id, row.baseline_record_id
            )
            if previous_baseline != row.baseline_record_id:
                raise ValueError(
                    f"multiple all-control baselines for wall {row.wall_id}"
                )
            positions = by_wall[row.wall_id]
            if row.table_position in positions:
                raise ValueError(
                    "duplicate table position "
                    f"{row.table_position} for wall {row.wall_id}"
                )
            positions[row.table_position] = row

        complete: dict[str, list[float]] = {}
        incomplete: list[str] = []
        for wall_id, positions in sorted(by_wall.items()):
            if set(positions) != set(range(4)):
                incomplete.append(wall_id)
                continue
            complete[wall_id] = [positions[index].score_delta for index in range(4)]
        if incomplete and require_complete_walls:
            raise ValueError(
                "matched-control reduction requires every wall in all four chairs; "
                f"incomplete walls: {', '.join(incomplete)}"
            )
        if not complete:
            raise ValueError("no complete four-chair matched-control walls")

        reductions.append(
            MatchedControlReduction(
                profile=profile,
                control=control,
                model=model,
                score_delta=cluster_bootstrap_mean(
                    complete,
                    confidence=confidence,
                    samples=bootstrap_samples,
                    seed=seed,
                ),
                block_means=tuple(
                    (wall_id, fmean(values))
                    for wall_id, values in sorted(complete.items())
                ),
                incomplete_walls=tuple(incomplete),
            )
        )
    return reductions
