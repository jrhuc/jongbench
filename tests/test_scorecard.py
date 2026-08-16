from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from jongbench import scorecard


def _state(**overrides) -> SimpleNamespace:
    values = {
        "riichi_accepted": [False, False, False, False],
        "player_id": 0,
        "last_cans": SimpleNamespace(can_discard=True),
        "real_time_shanten": lambda: 2,
        "at_furiten": False,
        "tiles_left": 40,
        "is_all_last": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_competence_tags_mark_pushfold_and_defense() -> None:
    menu = [{"kind": "discard", "label": "discard 1m"}]
    events = [{"type": "start_kyoku"}]
    assert "efficiency" in scorecard.competence_tags(_state(), menu, events)

    riichi = _state(riichi_accepted=[False, True, False, False])
    assert "pushfold" in scorecard.competence_tags(riichi, menu, events)

    nonzero_seat = _state(
        player_id=2, riichi_accepted=[True, False, False, False]
    )
    assert "pushfold" in scorecard.competence_tags(nonzero_seat, menu, events)
    deep = _state(
        riichi_accepted=[False, True, False, False],
        real_time_shanten=lambda: 3,
    )
    tags = scorecard.competence_tags(deep, menu, events)
    assert "defense_only" in tags
    assert "pushfold" in tags


def test_competence_tags_mark_calls_riichi_and_endgame() -> None:
    menu = [{"kind": "riichi"}, {"kind": "chi"}, {"kind": "ankan"}]
    state = _state(
        last_cans=SimpleNamespace(can_discard=False),
        at_furiten=True,
        tiles_left=4,
        is_all_last=True,
    )
    tags = scorecard.competence_tags(state, menu, [{"type": "start_kyoku"}])
    assert tags == [
        "riichi_choice",
        "call_choice",
        "oorasu_placement",
        "kan_choice",
        "furiten",
        "last_turns",
    ]


class _AnalysisState:
    is_menzen = True

    def __init__(self, analyses):
        self.analyses = analyses

    def reaction_analysis(self, event_json: str):
        event = json.loads(event_json)
        return self.analyses[event["pai"]]


def _discard(tile: str) -> dict:
    return {"kind": "discard", "event": {"type": "dahai", "actor": 0, "pai": tile}}


def test_ukeire_compares_only_alternatives_at_the_best_shanten() -> None:
    state = _AnalysisState(
        {
            "1m": (1, [], False, 5, True, True),
            "2m": (2, [], False, 20, True, True),
        }
    )
    menu = [_discard("1m"), _discard("2m")]
    best = scorecard.analyze_choice(state, menu, menu[0]["event"])
    assert best["ukeire_loss"] == 0
    assert not best["needless_shanten_regression"]

    regression = scorecard.analyze_choice(state, menu, menu[1]["event"])
    assert regression["ukeire_loss"] == 0
    assert regression["needless_shanten_regression"]


def test_same_shanten_ukeire_loss_is_exact() -> None:
    state = _AnalysisState(
        {
            "1m": (1, [], False, 5, True, True),
            "2m": (1, [], False, 12, True, True),
        }
    )
    menu = [_discard("1m"), _discard("2m")]
    analysis = scorecard.analyze_choice(state, menu, menu[0]["event"])
    assert analysis["ukeire_loss"] == 7
    assert not analysis["needless_shanten_regression"]


def test_yakuless_flag_means_unriichiable_ronless_closed_tenpai() -> None:
    state = _AnalysisState({"1m": (0, [3], False, 4, False, True)})
    discard = _discard("1m")
    assert scorecard.analyze_choice(state, [discard], discard["event"])[
        "yakuless_tenpai"
    ]
    riichi = {"kind": "riichi", "event": {"type": "reach", "actor": 0}}
    assert not scorecard.analyze_choice(
        state, [discard, riichi], discard["event"]
    )["yakuless_tenpai"]


def test_exact_scorecard_refuses_an_inexact_fallback() -> None:
    state = SimpleNamespace(is_menzen=True)
    with pytest.raises(RuntimeError, match="reaction_analysis"):
        scorecard.analyze_choice(state, [_discard("1m")], _discard("1m")["event"])
