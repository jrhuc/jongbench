from __future__ import annotations

import pytest

from jongbench import competence


def test_fingerprint_counts_hands_calls_and_kyoku_deltas() -> None:
    events = [
        {"type": "start_kyoku"},
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "dahai", "actor": 0, "pai": "1m"},
        {"type": "chi", "actor": 0},
        {"type": "hora", "actor": 0, "target": 1, "deltas": [8000, -8000, 0, 0]},
        {"type": "end_kyoku"},
        {"type": "start_kyoku"},
        {"type": "reach", "actor": 1},
        {"type": "reach_accepted", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "2m"},
        {"type": "tsumo", "actor": 0, "pai": "3m"},
        {"type": "dahai", "actor": 0, "pai": "2m"},
        {
            "type": "ryukyoku",
            "deltas": [1500, -1500, 0, 0],
            "tenpais": [True, False, False, False],
        },
        {"type": "end_kyoku"},
    ]
    fp = competence.behavioral_fingerprint(events, 0)
    assert fp["kyoku"] == 2
    assert fp["win_rate"] == 0.5
    assert fp["call_actions_per_hand"] == 0.5
    assert fp["open_hand_rate"] == 0.5
    assert fp["tenpai_at_draw_rate"] == 1.0
    assert fp["avg_kyoku_point_delta"] == pytest.approx(4750)
    assert fp["all_riichi_genbutsu_rate"] == 1.0
    assert fp["fold_rate"] == fp["all_riichi_genbutsu_rate"]



def test_draw_tenpai_uses_revealed_hands_even_when_deltas_are_zero() -> None:
    events = [
        {"type": "start_kyoku"},
        {
            "type": "ryukyoku",
            "deltas": [0, 0, 0, 0],
            "tehais": [["1m"] * 13, ["2m"] * 13, ["3m"] * 13, ["4m"] * 13],
        },
        {"type": "end_kyoku"},
    ]
    assert competence.behavioral_fingerprint(events, 0)["tenpai_at_draw_rate"] == 1.0

def test_fingerprint_counts_double_ron_once_per_hand() -> None:
    events = [
        {"type": "start_kyoku"},
        {"type": "dahai", "actor": 0, "pai": "9m"},
        {"type": "hora", "actor": 1, "target": 0, "deltas": [-8000, 8000, 0, 0]},
        {"type": "hora", "actor": 2, "target": 0, "deltas": [-3900, 0, 3900, 0]},
        {"type": "end_kyoku"},
    ]
    fp = competence.behavioral_fingerprint(events, 0)
    assert fp["deal_in_rate"] == 1.0
    assert fp["avg_deal_in_value"] == pytest.approx(11900)
    assert fp["avg_kyoku_point_delta"] == pytest.approx(-11900)


def test_fold_metric_requires_safety_against_every_riichi_opponent() -> None:
    events = [
        {"type": "start_kyoku"},
        {"type": "dahai", "actor": 1, "pai": "1m"},
        {"type": "reach_accepted", "actor": 1},
        {"type": "dahai", "actor": 2, "pai": "2m"},
        {"type": "reach_accepted", "actor": 2},
        {"type": "dahai", "actor": 0, "pai": "1m"},
        {"type": "dahai", "actor": 2, "pai": "1m"},
        {"type": "dahai", "actor": 0, "pai": "1m"},
        {"type": "end_kyoku"},
    ]
    fp = competence.behavioral_fingerprint(events, 0)
    assert fp["all_riichi_genbutsu_rate"] == pytest.approx(0.5)


def test_style_delta_uses_reviewer_distribution_and_q_loss_exposes_scale() -> None:
    review = {
        "entries": [
            {
                "actual": {"type": "dahai"},
                "expected": {"type": "reach"},
                "actual_index": 0,
                "details": [
                    {"event": {"type": "dahai"}, "q_value": 1.0, "prob": 0.25},
                    {"event": {"type": "reach"}, "q_value": 3.0, "prob": 0.75},
                ],
            },
            {
                "actual": {"type": "dahai"},
                "expected": {"type": "dahai"},
                "actual_index": 1,
                "details": [
                    {"event": {"type": "dahai"}, "q_value": 2.0, "prob": 0.5},
                    {"event": {"type": "dahai"}, "q_value": 2.0, "prob": 0.5},
                ],
            },
        ]
    }
    delta = competence.style_delta(review)
    assert delta["discard"] == pytest.approx(0.375)
    assert delta["riichi"] == pytest.approx(-0.375)
    loss = competence.cumulative_q_loss(review)
    assert loss["q_loss"] == pytest.approx(2.0)
    assert loss["q_loss_per_decision"] == pytest.approx(1.0)
    assert loss["normalised_q_loss_per_decision"] == pytest.approx(0.5)
    assert loss["mean_q_span"] == pytest.approx(1.0)
    assert competence.q_loss_of_choice([1.0, 3.0], 0) == pytest.approx(2.0)


def test_style_delta_falls_back_to_expected_action_for_old_reviews() -> None:
    review = {
        "entries": [
            {
                "actual": {"type": "dahai"},
                "expected": {"type": "reach"},
                "actual_index": 0,
                "details": [{"q_value": 1.0}, {"q_value": 3.0}],
            }
        ]
    }
    delta = competence.style_delta(review)
    assert delta["discard"] == 1.0
    assert delta["riichi"] == -1.0


def test_calibrate_q_loss_fits_a_line_and_reports_error() -> None:
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
    assert fit["rmse"] == pytest.approx(0.0)


def test_zero_delta_legacy_draw_is_unknown_not_noten() -> None:
    events = [
        {"type": "start_kyoku"},
        {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
        {"type": "end_kyoku"},
    ]
    fp = competence.behavioral_fingerprint(events, 0)
    assert fp["tenpai_at_draw_rate"] == 0.0
    assert fp["tenpai_at_draw_coverage"] == 0.0
