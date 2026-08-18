"""Statistics for rotated hanchan and paired-control comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from statistics import stdev
from typing import Any


DUPLICATE_BLOCK = 4
SEED_BASE = 20260000


def episode_deal(*, idx: int, seat_rotation: bool = True) -> dict[str, int]:
    """Map an episode index onto a wall seed and chair rotation."""
    if seat_rotation:
        return {
            "seed": SEED_BASE + idx // DUPLICATE_BLOCK,
            "rotation": idx % DUPLICATE_BLOCK,
            "block": idx // DUPLICATE_BLOCK,
        }
    return {"seed": SEED_BASE + idx, "rotation": 0, "block": idx}


def table_score_margin(own: float, others: Sequence[float]) -> float:
    """Own final score minus the mean of the other seats at the same table.

    This is descriptive, not luck-adjusted.  At a standard constant-sum table it is
    an affine transform of ``own`` and therefore cannot add information or cancel deal
    luck.  Use :func:`paired_control_delta` for a genuine same-seed counterfactual.
    """
    if not others:
        raise ValueError("table score margin needs at least one opposing score")
    return float(own) - (sum(float(score) for score in others) / len(others))


def score_differential(own: float, others: Sequence[float]) -> float:
    """Compatibility alias for :func:`table_score_margin`.

    The old name suggested a duplicate-adjusted quantity; it is only a within-table
    margin and should not be published as an independent outcome signal.
    """
    return table_score_margin(own, others)


def paired_control_delta(observed: float, all_control: float) -> float:
    """Score change from replacing the matched all-control seat with the policy.

    Both scores must refer to the same wall seed and table position.  Averaging the
    four chair deltas in a duplicate block estimates the intervention relative to the
    fixed control while retaining common random numbers.
    """
    return float(observed) - float(all_control)


def paired_control_stats(
    observed: Sequence[float],
    all_control: Sequence[float],
    *,
    block: int = DUPLICATE_BLOCK,
) -> dict[str, Any]:
    """Aggregate matched policy-replacement deltas by duplicate wall block."""
    if len(observed) != len(all_control):
        raise ValueError(
            "paired comparison requires the same number of policy and control scores"
        )
    return seed_block_stats(
        [
            paired_control_delta(policy_score, control_score)
            for policy_score, control_score in zip(
                observed, all_control, strict=True
            )
        ],
        block=block,
    )


def seed_block_stats(
    values: Sequence[float], *, block: int = DUPLICATE_BLOCK
) -> dict[str, Any]:
    """Mean of per-seed block means, with the standard error of those means."""
    if block <= 0:
        raise ValueError("block size must be positive")
    if len(values) % block != 0:
        raise ValueError(
            f"duplicate comparison requires a multiple of {block} episodes, "
            f"got {len(values)}"
        )
    if not values:
        raise ValueError("no values")
    numbers = [float(value) for value in values]
    block_means = [
        sum(numbers[start : start + block]) / block
        for start in range(0, len(numbers), block)
    ]
    mean = sum(block_means) / len(block_means)
    error = (
        stdev(block_means) / sqrt(len(block_means)) if len(block_means) > 1 else None
    )
    return {
        "mean": mean,
        "standard_error": error,
        "block_means": block_means,
        "n_blocks": len(block_means),
        "n": len(numbers),
    }
