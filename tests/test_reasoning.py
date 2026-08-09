from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jongbench import engines, reasoning


def _record(kyoku=0, honba=0, junme=1, tiles_left=69, label="discard 1m", why="", **extra):
    return {
        "player_id": 0,
        "kyoku": kyoku,
        "honba": honba,
        "junme": junme,
        "tiles_left": tiles_left,
        "choice_label": label,
        "raw_reasoning": why,
        **extra,
    }


def _entry(kyoku=0, honba=0, junme=1, tiles_left=69, actual_index=0, probs=(0.7, 0.3)):
    return {
        "kyoku": kyoku,
        "honba": honba,
        "junme": junme,
        "tiles_left": tiles_left,
        "actual_index": actual_index,
        "is_equal": actual_index == 0,
        "shanten": 2,
        "expected": {"type": "dahai", "pai": "1m"},
        "actual": {"type": "dahai", "pai": "9p"},
        "details": [{"prob": p, "q_value": -p} for p in probs],
    }


def test_prob_loss_is_the_gap_to_mortals_favourite() -> None:
    assert reasoning.prob_loss(_entry(actual_index=1, probs=(0.7, 0.3))) == pytest.approx(0.4)
    assert reasoning.prob_loss(_entry(actual_index=0, probs=(0.7, 0.3))) == 0.0
    assert reasoning.prob_loss({"details": [], "actual_index": 0}) == 0.0
    assert reasoning.prob_loss({"details": [{"prob": 0.5}], "actual_index": 9}) == 0.0


def test_join_is_by_board_position_not_by_order() -> None:
    """The logs are different lengths - a furo-toggle pass never reached the model, and a
    forced action was never graded - so a positional zip would silently mismatch."""
    decisions = [
        _record(junme=1, tiles_left=69, why="a"),
        _record(junme=3, tiles_left=65, why="c"),
    ]
    review = {
        "entries": [
            _entry(junme=1, tiles_left=69, actual_index=0),
            _entry(junme=2, tiles_left=67, actual_index=1),  # never asked of the model
            _entry(junme=3, tiles_left=65, actual_index=1, probs=(0.9, 0.1)),
        ]
    }

    joined = reasoning.join(decisions, review)

    assert [d.reasoning for d in joined.decisions] == ["a", "c"]
    assert joined.decisions[0].prob_loss == 0.0
    assert joined.decisions[1].prob_loss == pytest.approx(0.8)
    assert joined.graded == 3
    assert joined.logged == 2
    assert joined.unjoined_logged == 0


def test_repeated_coordinates_pair_up_in_order() -> None:
    """Riichi is two decisions at one coordinate: the declaration and the discard."""
    decisions = [_record(why="declare"), _record(why="then discard")]
    review = {"entries": [_entry(actual_index=0), _entry(actual_index=1)]}

    joined = reasoning.join(decisions, review)

    assert [d.reasoning for d in joined.decisions] == ["declare", "then discard"]
    assert [d.is_equal for d in joined.decisions] == [True, False]
    assert joined.graded == 2


def test_a_decision_mortal_never_graded_is_reported_not_dropped_silently() -> None:
    joined = reasoning.join([_record(junme=7)], {"entries": [_entry(junme=1)]})
    assert joined.decisions == []
    assert joined.logged == 1
    assert joined.unjoined_logged == 1
    assert joined.coverage == 0.0


def test_summary_contrasts_reasoning_length_against_agreement() -> None:
    decisions = [
        _record(junme=1, tiles_left=69, why="x" * 100),
        _record(junme=2, tiles_left=68, why="y" * 10),
    ]
    review = {
        "entries": [
            _entry(junme=1, tiles_left=69, actual_index=0),
            _entry(junme=2, tiles_left=68, actual_index=1),
        ]
    }

    summary = reasoning.join(decisions, review).summary()

    assert summary["joined"] == 2
    assert summary["coverage"] == 1.0
    assert summary["with_reasoning"] == 2
    assert summary["mean_reasoning_chars_when_matching_mortal"] == 100
    assert summary["mean_reasoning_chars_when_not"] == 10


def test_worst_ranks_by_probability_lost() -> None:
    decisions = [_record(junme=j, tiles_left=70 - j, why=str(j)) for j in (1, 2, 3)]
    review = {
        "entries": [
            _entry(junme=1, tiles_left=69, actual_index=1, probs=(0.5, 0.4)),
            _entry(junme=2, tiles_left=68, actual_index=1, probs=(0.9, 0.05)),
            _entry(junme=3, tiles_left=67, actual_index=0),
        ]
    }

    worst = reasoning.join(decisions, review).worst(2)

    assert [d.reasoning for d in worst] == ["2", "1"]


def test_engine_coordinates_match_the_shape_review_uses() -> None:
    events = [
        {"type": "start_kyoku", "bakaze": "S", "kyoku": 3, "honba": 2},
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "dahai", "actor": 0, "pai": "1m"},
        {"type": "tsumo", "actor": 1, "pai": "2m"},
        {"type": "dahai", "actor": 1, "pai": "2m"},
        {"type": "pon", "actor": 0, "pai": "2m"},
    ]
    coords = engines._decision_coords(0, events)
    assert coords["kyoku"] == 4 + 3 - 1  # South 3 is the seventh hand
    assert coords["honba"] == 2
    assert coords["junme"] == 2  # own tsumo, then own pon
    assert coords["tiles_left"] == 68  # two draws off 70
