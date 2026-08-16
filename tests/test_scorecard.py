from __future__ import annotations

from types import SimpleNamespace

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
