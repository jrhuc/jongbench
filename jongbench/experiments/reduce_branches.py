"""Reducer for exact-hidden-wall action interventions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean
from typing import Iterable

from .identity import ControlPolicyIdentity
from .records_branch import BranchResult
from .statistics import ClusterEstimate, cluster_bootstrap_mean


@dataclass(frozen=True, slots=True)
class BranchDecisionEstimate:
    opportunity_id: str
    decision_id: str
    wall_id: str
    profile: str
    seat: str
    action_means: tuple[tuple[int, str, float, int], ...]
    model_choice: int
    reference_choice: int | None
    complete_action_set: bool
    hindsight_best_action: int | None
    hindsight_regret: float | None
    model_vs_reference_delta: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "opportunity_id": self.opportunity_id,
            "decision_id": self.decision_id,
            "wall_id": self.wall_id,
            "profile": self.profile,
            "seat": self.seat,
            "action_means": [
                {
                    "action_index": index,
                    "action_label": label,
                    "mean_point_delta": mean,
                    "n": count,
                }
                for index, label, mean, count in self.action_means
            ],
            "model_choice": self.model_choice,
            "reference_choice": self.reference_choice,
            "complete_action_set": self.complete_action_set,
            "hindsight_best_action": self.hindsight_best_action,
            "hindsight_regret": self.hindsight_regret,
            "model_vs_reference_delta": self.model_vs_reference_delta,
        }


@dataclass(frozen=True, slots=True)
class BranchReduction:
    profile: str
    control: ControlPolicyIdentity
    estimand: str
    decisions: tuple[BranchDecisionEstimate, ...]
    hindsight_regret: ClusterEstimate | None
    model_vs_reference_delta: ClusterEstimate | None
    incomplete_decisions: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "record_type": "branch_reduction",
            "profile": self.profile,
            "control": self.control.as_dict(),
            "estimand": self.estimand,
            "decisions": [decision.as_dict() for decision in self.decisions],
            "hindsight_regret": (
                None
                if self.hindsight_regret is None
                else self.hindsight_regret.as_dict()
            ),
            "model_vs_reference_delta": (
                None
                if self.model_vs_reference_delta is None
                else self.model_vs_reference_delta.as_dict()
            ),
            "incomplete_decisions": list(self.incomplete_decisions),
        }


def _reduce_decision(rows: list[BranchResult]) -> BranchDecisionEstimate:
    first = rows[0]
    invariant = (
        first.opportunity_id,
        first.wall_id,
        first.profile,
        first.control,
        first.estimand,
        first.seat,
        first.action_count,
        first.model_choice,
        first.reference_choice,
    )
    for row in rows[1:]:
        actual = (
            row.opportunity_id,
            row.wall_id,
            row.profile,
            row.control,
            row.estimand,
            row.seat,
            row.action_count,
            row.model_choice,
            row.reference_choice,
        )
        if actual != invariant:
            raise ValueError(f"inconsistent branch rows for {first.decision_id}")

    by_action: dict[int, list[BranchResult]] = defaultdict(list)
    for row in rows:
        by_action[row.action_index].append(row)
    duplicates = [
        action for action, action_rows in by_action.items() if len(action_rows) > 1
    ]
    if duplicates:
        raise ValueError(
            "factual-wall records allow exactly one result per action; duplicates: "
            + ", ".join(map(str, sorted(duplicates)))
        )

    action_means = tuple(
        (
            action,
            action_rows[0].action_label,
            fmean(item.point_delta for item in action_rows),
            len(action_rows),
        )
        for action, action_rows in sorted(by_action.items())
    )
    means = {action: mean for action, _, mean, _ in action_means}
    complete = set(means) == set(range(first.action_count))
    best_action: int | None = None
    regret: float | None = None
    if complete and first.model_choice in means:
        best_action = max(means, key=lambda action: (means[action], -action))
        regret = means[best_action] - means[first.model_choice]

    reference_delta: float | None = None
    if (
        first.reference_choice is not None
        and first.model_choice in means
        and first.reference_choice in means
    ):
        reference_delta = means[first.model_choice] - means[first.reference_choice]

    return BranchDecisionEstimate(
        opportunity_id=first.opportunity_id,
        decision_id=first.decision_id,
        wall_id=first.wall_id,
        profile=first.profile,
        seat=first.seat,
        action_means=action_means,
        model_choice=first.model_choice,
        reference_choice=first.reference_choice,
        complete_action_set=complete,
        hindsight_best_action=best_action,
        hindsight_regret=regret,
        model_vs_reference_delta=reference_delta,
    )


def reduce_factual_branches(
    rows: Iterable[BranchResult],
    *,
    confidence: float = 0.95,
    bootstrap_samples: int = 4000,
    seed: int = 0,
) -> list[BranchReduction]:
    """Reduce exact-wall interventions without calling them expected values."""
    grouped: dict[
        tuple[str, ControlPolicyIdentity, str], dict[str, list[BranchResult]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(row.profile, row.control, row.estimand)][row.decision_id].append(row)
    if not grouped:
        raise ValueError("no branch rows")

    reductions: list[BranchReduction] = []
    for (profile, control, estimand), decision_rows in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1].policy_id, item[0][2]),
    ):
        decisions = tuple(
            _reduce_decision(group) for _, group in sorted(decision_rows.items())
        )
        regret_clusters: dict[str, list[float]] = defaultdict(list)
        reference_clusters: dict[str, list[float]] = defaultdict(list)
        for decision in decisions:
            if decision.hindsight_regret is not None:
                regret_clusters[decision.wall_id].append(decision.hindsight_regret)
            if decision.model_vs_reference_delta is not None:
                reference_clusters[decision.wall_id].append(
                    decision.model_vs_reference_delta
                )
        reductions.append(
            BranchReduction(
                profile=profile,
                control=control,
                estimand=estimand,
                decisions=decisions,
                hindsight_regret=(
                    cluster_bootstrap_mean(
                        regret_clusters,
                        confidence=confidence,
                        samples=bootstrap_samples,
                        seed=seed,
                    )
                    if regret_clusters
                    else None
                ),
                model_vs_reference_delta=(
                    cluster_bootstrap_mean(
                        reference_clusters,
                        confidence=confidence,
                        samples=bootstrap_samples,
                        seed=seed,
                    )
                    if reference_clusters
                    else None
                ),
                incomplete_decisions=tuple(
                    decision.decision_id
                    for decision in decisions
                    if not decision.complete_action_set
                ),
            )
        )
    return reductions
