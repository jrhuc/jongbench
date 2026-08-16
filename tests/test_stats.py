from __future__ import annotations

import pytest

from jongbench import stats


def test_episode_deal_uses_duplicate_blocks() -> None:
    assert stats.episode_deal(idx=0) == {
        "seed": stats.SEED_BASE,
        "rotation": 0,
        "block": 0,
    }
    assert stats.episode_deal(idx=3) == {
        "seed": stats.SEED_BASE,
        "rotation": 3,
        "block": 0,
    }
    assert stats.episode_deal(idx=4) == {
        "seed": stats.SEED_BASE + 1,
        "rotation": 0,
        "block": 1,
    }


def test_episode_deal_without_rotation_keeps_unique_walls() -> None:
    assert stats.episode_deal(idx=3, seat_rotation=False) == {
        "seed": stats.SEED_BASE + 3,
        "rotation": 0,
        "block": 3,
    }


def test_score_differential_is_own_minus_opponents_mean() -> None:
    assert stats.score_differential(40000, [25000, 20000, 15000]) == pytest.approx(
        20000
    )


def test_seed_block_stats_require_a_multiple_of_four() -> None:
    with pytest.raises(ValueError, match="multiple of 4"):
        stats.seed_block_stats([1.0, 0.0, 1.0])
    result = stats.seed_block_stats([1.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5])
    assert result["n_blocks"] == 2
    assert result["mean"] == pytest.approx(0.5)
    assert result["standard_error"] == pytest.approx(0.0)
