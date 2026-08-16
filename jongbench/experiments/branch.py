"""Exact-hidden-wall branch audits for recorded decisions."""

from __future__ import annotations

import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .. import arena
from ..artifacts import load_mjai_log
from ..controls import MortalControlPool
from ..weights import AUTO_MORTAL_WEIGHTS, CheckpointInput
from .capsule import ReplayCapsule, ScriptedDecision
from .execution import (
    control_identity,
    control_pool,
    fresh_log_dir,
    only_log,
    summary_for,
)
from .observations import BranchResult, write_jsonl
from .runtime import ReplayDiverged, ReplayThenEngine

_BAKAZE_ORDER = {"E": 0, "S": 1, "W": 2, "N": 3}


def _branch_table(
    capsule: ReplayCapsule,
    scripts: dict[str, tuple[ScriptedDecision, ...]],
    pool: MortalControlPool,
    replay_until: tuple[int, int, int],
) -> tuple[list[ReplayThenEngine], list[ReplayThenEngine]]:
    wrappers = {
        seat: ReplayThenEngine(
            seat,
            scripts[seat],
            pool.make_arena(seat),
            replay_until=replay_until,
            scripted_policy=capsule.model(seat) != "mortal",
        )
        for seat in capsule.table
    }
    return [wrappers[seat] for seat in capsule.table], list(wrappers.values())


def _absolute_kyoku(event: dict[str, Any]) -> int:
    return (
        _BAKAZE_ORDER.get(str(event.get("bakaze")), 0) * 4
        + int(event.get("kyoku", 1))
        - 1
    )


def factual_kyoku_point_delta(
    events: Sequence[dict[str, Any]],
    *,
    player_id: int,
    kyoku: int,
    honba: int,
) -> float:
    """Realized point delta in one exact hidden-wall branch."""
    active = False
    found = False
    total = 0.0
    for event in events:
        event_type = event.get("type")
        if event_type == "start_kyoku":
            active = (
                _absolute_kyoku(event) == kyoku
                and int(event.get("honba", 0)) == honba
            )
            found |= active
            continue
        if not active:
            continue
        if event_type in {"hora", "ryukyoku"}:
            deltas = event.get("deltas")
            if isinstance(deltas, list) and player_id < len(deltas):
                total += float(deltas[player_id])
        if event_type == "end_kyoku":
            return total
    if not found:
        raise ReplayDiverged(f"branched log never reached kyoku={kyoku}, honba={honba}")
    raise ReplayDiverged("branched log ended before the target kyoku completed")


def run_factual_branches(
    capsule: ReplayCapsule,
    target_sequence: int,
    *,
    choices: Iterable[int] | None = None,
    reference_choice: int | None = None,
    weights: CheckpointInput = AUTO_MORTAL_WEIGHTS,
    device: object | str | None = None,
    output_dir: str | Path | None = None,
    require_checkpoint_match: bool = True,
) -> list[BranchResult]:
    """Force legal actions on the episode's exact hidden wall.

    The output is causal for the realized wall. Selecting the best action after seeing
    that hidden wall is a hindsight audit, not an expected action value.
    """
    target = capsule.decision(target_sequence)
    selected = tuple(range(len(target.menu))) if choices is None else tuple(choices)
    if not selected:
        raise ValueError("at least one branch choice is required")
    if len(set(selected)) != len(selected):
        raise ValueError("branch choices contain duplicates")
    if any(choice not in range(len(target.menu)) for choice in selected):
        raise ValueError("branch choice is out of range")
    if reference_choice is not None and reference_choice not in range(len(target.menu)):
        raise ValueError("reference_choice is out of range")

    pool = control_pool(
        capsule,
        weights=weights,
        device=device,
        require_checkpoint_match=require_checkpoint_match,
    )
    control = control_identity(capsule, pool)
    persistent_root = Path(output_dir) if output_dir is not None else None
    if persistent_root is not None:
        persistent_root.mkdir(parents=True, exist_ok=True)

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if persistent_root is None:
        temporary = tempfile.TemporaryDirectory(prefix="jongbench-branch-")
        root = Path(temporary.name)
    else:
        root = persistent_root

    results: list[BranchResult] = []
    try:
        for choice in selected:
            branch_dir = (
                root / f"decision-{target_sequence:05d}" / f"action-{choice:02d}"
            )
            log_dir = fresh_log_dir(branch_dir / "logs")
            scripts = capsule.scripts_by_seat(target_sequence, choice)
            table, wrappers = _branch_table(
                capsule, scripts, pool, target.opportunity_key
            )
            summaries = arena.run_games(
                table,
                1,
                seed_start=capsule.seed,
                log_dir=str(log_dir),
                disable_progress_bar=True,
            )
            for wrapper in wrappers:
                wrapper.assert_consumed()
            log_path = only_log(log_dir)
            events = load_mjai_log(log_path)
            player_id = capsule.table_position(target.seat)
            point_delta = factual_kyoku_point_delta(
                events,
                player_id=player_id,
                kyoku=target.kyoku,
                honba=target.honba,
            )
            final_score, placement = summary_for(summaries[0], target.seat)
            results.append(
                BranchResult.create(
                    capsule_id=capsule.capsule_id,
                    opportunity_id=capsule.opportunity_id(target_sequence),
                    decision_id=capsule.decision_id(target_sequence),
                    wall_id=capsule.wall_id,
                    profile=capsule.profile,
                    control=control,
                    estimand="factual_wall",
                    target_sequence=target_sequence,
                    seat=target.seat,
                    table_position=player_id,
                    kyoku=target.kyoku,
                    honba=target.honba,
                    junme=target.junme,
                    action_index=choice,
                    action_count=len(target.menu),
                    action_label=target.menu[choice],
                    model_choice=target.choice,
                    reference_choice=reference_choice,
                    point_delta=point_delta,
                    final_score=final_score,
                    placement=placement,
                    log_path=(
                        str(log_path) if persistent_root is not None else None
                    ),
                )
            )
        if persistent_root is not None:
            write_jsonl(
                persistent_root
                / f"decision-{target_sequence:05d}"
                / "branch-results.jsonl",
                list(results),
            )
        return results
    finally:
        if temporary is not None:
            temporary.cleanup()
