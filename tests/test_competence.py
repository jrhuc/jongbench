from __future__ import annotations

import pytest

from jongbench import competence


def test_fingerprint_counts_wins_calls_and_kyoku_deltas() -> None:
    events = [
        {"type": "start_kyoku"},
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "dahai", "actor": 0, "pai": "1m"},
        {"type": "chi", "actor": 0},
        {"type": "hora", "actor": 0, "target": 1, "deltas": [8000, -8000, 0, 0]},
        {"type": "start_kyoku"},
        {"type": "reach", "actor": 1},
        {"type": "reach_accepted", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "2m"},
        {"type": "tsumo", "actor": 0, "pai": "3m"},
        {"type": "dahai", "actor": 0, "pai": "2m"},
        {"type": "ryukyoku", "deltas": [1500, -1500, 0, 0]},
    ]
    fp = competence.behavioral_fingerprint(events, 0)
    assert fp["kyoku"] == 2
    assert fp["win_rate"] == 0.5
    assert fp["call_rate"] == 0.5
    assert fp["tenpai_at_draw_rate"] == 1.0
    assert fp["avg_kyoku_point_delta"] == pytest.approx(4750)
    assert fp["fold_rate"] == 1.0


def test_style_delta_and_q_loss_are_reviewer_units() -> None:
    review = {
        "entries": [
            {
                "actual": {"type": "dahai"},
                "expected": {"type": "reach"},
                "actual_index": 0,
                "details": [{"q_value": 1.0}, {"q_value": 3.0}],
            },
            {
                "actual": {"type": "dahai"},
                "expected": {"type": "dahai"},
                "actual_index": 1,
                "details": [{"q_value": 2.0}, {"q_value": 2.0}],
            },
        ]
    }
    delta = competence.style_delta(review)
    assert delta["discard"] == pytest.approx(0.5)
    assert delta["riichi"] == pytest.approx(-0.5)
    loss = competence.cumulative_q_loss(review)
    assert loss["q_loss"] == pytest.approx(2.0)
    assert loss["q_loss_per_decision"] == pytest.approx(1.0)
    assert competence.q_loss_of_choice([1.0, 3.0], 0) == pytest.approx(2.0)


def test_calibrate_q_loss_fits_a_line() -> None:
    fit = competence.calibrate_q_loss(
        [
            {"q_loss": 0.0, "points": 10.0},
            {"q_loss": 2.0, "points": 4.0},
        ]
    )
    assert fit["n"] == 2
    assert fit["slope"] == pytest.approx(-3.0)
    assert fit["intercept"] == pytest.approx(10.0)
    assert fit["r2"] == pytest.approx(1.0)
