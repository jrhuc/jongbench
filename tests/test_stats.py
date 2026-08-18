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


def test_table_margin_is_not_an_independent_outcome_signal() -> None:
    scores = [40000, 25000, 20000, 15000]
    margin = stats.table_score_margin(scores[0], scores[1:])
    assert margin == pytest.approx(20000)
    total = sum(scores)
    assert margin == pytest.approx((4 * scores[0] - total) / 3)
    assert stats.score_differential(scores[0], scores[1:]) == margin


def test_paired_control_delta_uses_same_seed_baseline() -> None:
    assert stats.paired_control_delta(31200, 27600) == pytest.approx(3600)


def test_seed_block_stats_require_a_multiple_of_four() -> None:
    with pytest.raises(ValueError, match="multiple of 4"):
        stats.seed_block_stats([1.0, 0.0, 1.0])
    result = stats.seed_block_stats([1.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5])
    assert result["n_blocks"] == 2
    assert result["mean"] == pytest.approx(0.5)
    assert result["standard_error"] == pytest.approx(0.0)


def test_paired_control_stats_preserve_wall_blocks() -> None:
    result = stats.paired_control_stats(
        [30_000, 29_000, 31_000, 32_000, 25_000, 26_000, 24_000, 27_000],
        [25_000, 25_000, 25_000, 25_000, 25_000, 25_000, 25_000, 25_000],
    )
    assert result["block_means"] == pytest.approx([5_500, 500])
    assert result["mean"] == pytest.approx(3_000)
    with pytest.raises(ValueError, match="same number"):
        stats.paired_control_stats([1.0], [])
