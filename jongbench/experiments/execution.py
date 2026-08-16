"""Shared execution boundary for control-backed experiments."""

from __future__ import annotations

import shutil
from pathlib import Path

from .. import arena
from ..controls import MortalControlConfig, MortalControlPool
from ..weights import CheckpointInput
from .capsule import ReplayCapsule
from .identity import ControlPolicyIdentity
from .runtime import ReplayDiverged

BASELINE_NAMES = ("control0", "control1", "control2", "control3")


def control_pool(
    capsule: ReplayCapsule,
    *,
    weights: CheckpointInput,
    device: object | str | None,
    require_checkpoint_match: bool,
) -> MortalControlPool:
    if capsule.control_boltzmann_epsilon != 0.0:
        raise ValueError(
            "causal replay currently requires deterministic controls "
            "(control_boltzmann_epsilon must be zero)"
        )
    pool = MortalControlPool(
        MortalControlConfig(
            weights=weights,
            use_policy=capsule.control_use_policy,
            boltzmann_epsilon=0.0,
            boltzmann_temp=capsule.control_boltzmann_temp,
            device=device,
        )
    )
    expected = capsule.control_checkpoint_sha256
    if (
        require_checkpoint_match
        and expected is not None
        and pool.checkpoint.sha256 != expected
    ):
        raise ValueError(
            "control checkpoint mismatch: capsule used "
            f"{expected}, experiment runner resolved {pool.checkpoint.sha256}"
        )
    return pool


def control_identity(
    capsule: ReplayCapsule, pool: MortalControlPool
) -> ControlPolicyIdentity:
    return capsule.control_identity_for(pool.checkpoint.sha256)


def only_log(log_dir: Path) -> Path:
    logs = sorted([*log_dir.glob("*.json.gz"), *log_dir.glob("*.json")])
    if len(logs) != 1:
        raise RuntimeError(
            f"expected exactly one experiment log in {log_dir}, got {len(logs)}"
        )
    return logs[0]


def fresh_log_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def summary_for(summary: arena.GameSummary, seat: str) -> tuple[float, int]:
    scores = dict(zip(summary.names, summary.scores, strict=True))
    if seat not in scores or seat not in summary.placements:
        raise ReplayDiverged(f"summary does not contain logical seat {seat!r}")
    return float(scores[seat]), int(summary.placements[seat])
