"""Duplicate-block statistics for hanchan outcome comparisons."""

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


def score_differential(own: float, others: Sequence[float]) -> float:
    """Own final score minus the mean of the other seats at the same table."""
    if not others:
        raise ValueError("score differential needs at least one opposing score")
    return float(own) - (sum(float(score) for score in others) / len(others))


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
