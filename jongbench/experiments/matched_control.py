"""All-control baselines and same-wall, same-chair policy replacements."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .. import arena
from ..weights import AUTO_MORTAL_WEIGHTS, CheckpointInput
from .capsule import ReplayCapsule
from .execution import (
    BASELINE_NAMES,
    control_identity,
    control_pool,
    fresh_log_dir,
    only_log,
)
from .observations import AllControlBaseline, MatchedControlResult
from .runtime import ReplayDiverged


def run_control_baseline(
    capsule: ReplayCapsule,
    *,
    weights: CheckpointInput = AUTO_MORTAL_WEIGHTS,
    device: object | str | None = None,
    output_dir: str | Path | None = None,
    require_checkpoint_match: bool = True,
) -> AllControlBaseline:
    """Run or load one all-control outcome for the capsule's wall."""
    pool = control_pool(
        capsule,
        weights=weights,
        device=device,
        require_checkpoint_match=require_checkpoint_match,
    )
    control = control_identity(capsule, pool)
    baseline_key = AllControlBaseline.key_for(capsule.wall_id, control)

    persistent_root = Path(output_dir) if output_dir is not None else None
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if persistent_root is None:
        temporary = tempfile.TemporaryDirectory(prefix="jongbench-control-")
        baseline_dir = Path(temporary.name)
    else:
        short_id = baseline_key.removeprefix("sha256:")[:12]
        baseline_dir = persistent_root / f"baseline-{short_id}"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        cached = baseline_dir / "baseline.json"
        if cached.is_file():
            value = json.loads(cached.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise TypeError(f"cached baseline must be a JSON object: {cached}")
            baseline = AllControlBaseline.from_dict(value)
            if baseline.baseline_key != baseline_key:
                raise ValueError(f"cached baseline key mismatch: {cached}")
            return baseline

    try:
        log_dir = fresh_log_dir(baseline_dir / "logs")
        summaries = arena.run_games(
            pool.make_table(BASELINE_NAMES),
            1,
            seed_start=capsule.seed,
            log_dir=str(log_dir),
            disable_progress_bar=True,
        )
        summary = summaries[0]
        if tuple(summary.names) != BASELINE_NAMES:
            raise ReplayDiverged(
                f"all-control table returned unexpected names {summary.names!r}"
            )
        placements = tuple(int(summary.placements[name]) for name in BASELINE_NAMES)
        baseline = AllControlBaseline.create(
            wall_id=capsule.wall_id,
            control=control,
            scores=tuple(float(score) for score in summary.scores),
            placements=(placements[0], placements[1], placements[2], placements[3]),
            log_path=(str(only_log(log_dir)) if persistent_root is not None else None),
        )
        if persistent_root is not None:
            (baseline_dir / "baseline.json").write_text(
                json.dumps(
                    baseline.as_dict(), ensure_ascii=False, indent=2, sort_keys=True
                )
                + "\n",
                encoding="utf-8",
            )
        return baseline
    finally:
        if temporary is not None:
            temporary.cleanup()


def match_control(
    capsule: ReplayCapsule,
    baseline: AllControlBaseline,
    *,
    seat: str | None = None,
) -> MatchedControlResult:
    """Join one policy episode to a cached all-control wall baseline."""
    evaluated = seat or capsule.evaluated_seat
    if evaluated is None:
        raise ValueError("tournament capsules require an explicit seat")
    if evaluated not in capsule.table:
        raise ValueError(f"unknown evaluated seat {evaluated!r}")
    capsule.assert_single_policy_replacement(evaluated)
    if baseline.wall_id != capsule.wall_id:
        raise ValueError("baseline wall does not match capsule wall")
    expected_checkpoint = capsule.control_checkpoint_sha256
    if (
        expected_checkpoint is not None
        and baseline.control.checkpoint_sha256 != expected_checkpoint
    ):
        raise ValueError("baseline control checkpoint does not match capsule controls")
    expected_control = capsule.control_identity_for(
        expected_checkpoint or baseline.control.checkpoint_sha256
    )
    if baseline.control != expected_control:
        raise ValueError("baseline control policy does not match capsule controls")

    position = capsule.table_position(evaluated)
    observed_score = float(capsule.score(evaluated))
    control_score = baseline.score(position)
    return MatchedControlResult.create(
        capsule_id=capsule.capsule_id,
        baseline_record_id=baseline.record_id,
        wall_id=capsule.wall_id,
        profile=capsule.profile,
        control=baseline.control,
        model=capsule.model(evaluated),
        seat=evaluated,
        table_position=position,
        observed_score=observed_score,
        control_score=control_score,
        observed_placement=capsule.placement(evaluated),
        control_placement=baseline.placement(position),
        log_path=baseline.log_path,
    )


def run_matched_control(
    capsule: ReplayCapsule,
    *,
    seat: str | None = None,
    weights: CheckpointInput = AUTO_MORTAL_WEIGHTS,
    device: object | str | None = None,
    output_dir: str | Path | None = None,
    require_checkpoint_match: bool = True,
) -> MatchedControlResult:
    """Cache the wall baseline, then join this episode to its physical chair."""
    baseline = run_control_baseline(
        capsule,
        weights=weights,
        device=device,
        output_dir=output_dir,
        require_checkpoint_match=require_checkpoint_match,
    )
    result = match_control(capsule, baseline, seat=seat)
    if output_dir is not None:
        root = Path(output_dir) / "matched"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{result.record_id.removeprefix('sha256:')[:16]}.json"
        path.write_text(
            json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    return result
