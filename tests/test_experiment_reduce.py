from __future__ import annotations

from dataclasses import replace

import pytest

from jongbench.experiments.identity import ControlPolicyIdentity
from jongbench.experiments.observations import (
    AllControlBaseline,
    BranchResult,
    MatchedControlResult,
    PairedArmObservation,
)
from jongbench.experiments.reduce import (
    reduce_factual_branches,
    reduce_matched_controls,
    reduce_paired_arms,
)

CONTROL = ControlPolicyIdentity.create(
    checkpoint_sha256="a" * 64,
    use_policy=False,
    boltzmann_epsilon=0.0,
    boltzmann_temp=1.0,
)


def _branch(
    decision: str,
    wall: str,
    action: int,
    value: float,
    *,
    model_choice: int = 0,
    reference_choice: int | None = 1,
) -> BranchResult:
    return BranchResult.create(
        capsule_id=f"capsule-{wall}",
        opportunity_id=f"opportunity-{decision}",
        decision_id=decision,
        wall_id=wall,
        profile="profile-a",
        control=CONTROL,
        estimand="factual_wall",
        target_sequence=3,
        seat="seat0",
        table_position=0,
        kyoku=0,
        honba=0,
        junme=4,
        action_index=action,
        action_count=2,
        action_label=f"action {action}",
        model_choice=model_choice,
        reference_choice=reference_choice,
        point_delta=value,
        final_score=25000 + value,
        placement=2,
    )


def test_factual_branch_reducer_keeps_hindsight_and_reference_separate() -> None:
    rows = [
        _branch("d1", "w1", 0, 0),
        _branch("d1", "w1", 1, 1000),
        _branch("d2", "w2", 0, 500),
        _branch("d2", "w2", 1, 0),
    ]
    reduction = reduce_factual_branches(rows, bootstrap_samples=100, seed=7)[0]

    assert reduction.estimand == "factual_wall"
    assert reduction.control == CONTROL
    assert reduction.hindsight_regret is not None
    assert reduction.hindsight_regret.mean == pytest.approx(500)
    assert reduction.model_vs_reference_delta is not None
    assert reduction.model_vs_reference_delta.mean == pytest.approx(-250)
    assert reduction.incomplete_decisions == ()


def test_factual_branch_reducer_rejects_duplicate_exact_wall_actions() -> None:
    rows = [_branch("d1", "w1", 0, 0), _branch("d1", "w1", 0, 0)]
    with pytest.raises(ValueError, match="exactly one result per action"):
        reduce_factual_branches(rows)


def _matched(wall: str, position: int, delta: float) -> MatchedControlResult:
    baseline = AllControlBaseline.create(
        wall_id=wall,
        control=CONTROL,
        scores=(25000.0, 25000.0, 25000.0, 25000.0),
        placements=(1, 2, 3, 4),
    )
    return MatchedControlResult.create(
        capsule_id=f"capsule-{wall}-{position}",
        baseline_record_id=baseline.record_id,
        wall_id=wall,
        profile="profile-a",
        control=CONTROL,
        model="model/a",
        seat="seat0",
        table_position=position,
        observed_score=25000 + delta,
        control_score=25000,
        observed_placement=2,
        control_placement=2,
    )


def test_matched_control_reducer_requires_complete_chair_blocks() -> None:
    rows = [
        *[_matched("w1", position, float(position)) for position in range(4)],
        *[_matched("w2", position, 10.0 + position) for position in range(4)],
    ]
    reduction = reduce_matched_controls(rows, bootstrap_samples=100, seed=3)[0]
    assert reduction.control == CONTROL
    assert reduction.score_delta.mean == pytest.approx(6.5)
    assert reduction.score_delta.n_clusters == 2
    assert reduction.incomplete_walls == ()

    with pytest.raises(ValueError, match="incomplete walls"):
        reduce_matched_controls(rows[:-1])


def _arm(
    pair: str, cluster: str, arm: str, value: float
) -> PairedArmObservation:
    return PairedArmObservation.create(
        pair_id=pair,
        cluster_id=cluster,
        profile="profile-a",
        experiment="hints",
        arm=arm,
        metric="reward",
        value=value,
    )


def test_paired_arm_reducer_never_pools_unpaired_variants() -> None:
    rows = [
        _arm("p1", "g1", "off", 0.4),
        _arm("p1", "g1", "on", 0.7),
        _arm("p2", "g2", "off", 0.5),
        _arm("p2", "g2", "on", 0.6),
        _arm("p3", "g3", "off", 0.9),
    ]
    reduction = reduce_paired_arms(
        rows,
        baseline_arm="off",
        treatment_arm="on",
        bootstrap_samples=100,
        seed=11,
    )[0]
    assert reduction.difference.mean == pytest.approx(0.2)
    assert reduction.complete_pairs == 2
    assert reduction.dropped_pairs == ("p3",)


def test_experiment_jsonl_round_trip(tmp_path) -> None:
    from jongbench.experiments.observations import read_jsonl, write_jsonl

    baseline = AllControlBaseline.create(
        wall_id="wall-a",
        control=CONTROL,
        scores=(25000.0, 25000.0, 25000.0, 25000.0),
        placements=(1, 2, 3, 4),
    )
    rows = [baseline, _branch("d1", "w1", 0, 123.0)]
    path = write_jsonl(tmp_path / "rows.jsonl", rows)
    assert read_jsonl(path) == rows


def test_control_identity_rejects_a_tampered_policy_id() -> None:
    with pytest.raises(ValueError, match="canonical content"):
        ControlPolicyIdentity(
            policy_id="sha256:" + "0" * 64,
            checkpoint_sha256="a" * 64,
            use_policy=False,
            boltzmann_epsilon=0.0,
            boltzmann_temp=1.0,
        )


def test_result_records_reject_tampered_content_ids() -> None:
    baseline = AllControlBaseline.create(
        wall_id="wall-a",
        control=CONTROL,
        scores=(25000.0, 25000.0, 25000.0, 25000.0),
        placements=(1, 2, 3, 4),
    )
    with pytest.raises(ValueError, match="record_id"):
        replace(baseline, scores=(26000.0, 24000.0, 25000.0, 25000.0))

    branch = _branch("d1", "w1", 0, 123.0)
    with pytest.raises(ValueError, match="result_id"):
        replace(branch, point_delta=456.0)

    paired = _arm("p1", "g1", "off", 0.4)
    with pytest.raises(ValueError, match="observation_id"):
        replace(paired, value=0.8)
