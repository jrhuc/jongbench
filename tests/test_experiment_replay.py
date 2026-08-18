from __future__ import annotations

from types import SimpleNamespace

import pytest

from jongbench import engines
from jongbench.experiments.branch import factual_kyoku_point_delta
from jongbench.experiments.capsule import ScriptedDecision
from jongbench.experiments.runtime import ReplayDiverged, ReplayThenEngine


class FakeContinuation(engines.BaseEngine):
    def __init__(self, name: str = "seat0") -> None:
        super().__init__(name, concurrency=1)
        self.decisions = 0
        self.auto_reactions = 0
        self.started: list[int] = []
        self.ended: list[int] = []

    def decide(self, player_id, state, events, menu, game_index=0):
        del player_id, state, events, game_index
        self.decisions += 1
        return menu[-1]["event"]

    def auto_reaction(self, state, menu, events, game_index):
        del state, events, game_index
        self.auto_reactions += 1
        return menu[-1]["event"]

    def start_game(self, game_idx: int) -> None:
        self.started.append(game_idx)

    def end_game(self, game_idx: int, scores: list[int]) -> None:
        del scores
        self.ended.append(game_idx)


def _decision(menu=("discard 1m", "discard 2m"), choice=1) -> ScriptedDecision:
    return ScriptedDecision(
        sequence=0,
        seat="seat0",
        player_id=0,
        kyoku=0,
        honba=0,
        junme=1,
        tiles_left=69,
        event_count=2,
        menu=menu,
        choice=choice,
        choice_label=menu[choice],
    )


def _menu(*labels: str, kinds: tuple[str, ...] | None = None):
    kinds = kinds or tuple("discard" for _ in labels)
    return [
        {"label": label, "kind": kind, "event": {"type": label}}
        for label, kind in zip(labels, kinds, strict=True)
    ]


def _events(count: int) -> list[dict]:
    if count < 1:
        raise ValueError("count must be positive")
    return [
        {"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0},
        *({"type": "dora"} for _ in range(count - 1)),
    ]


def test_replay_then_engine_has_one_explicit_handoff() -> None:
    continuation = FakeContinuation()
    wrapper = ReplayThenEngine("seat0", [_decision()], continuation)
    wrapper.set_player_ids([0])
    wrapper.start_game(0)

    menu = _menu("discard 1m", "discard 2m")
    assert wrapper.decide(0, None, _events(2), menu) == menu[1]["event"]
    wrapper.assert_consumed()
    assert wrapper.remaining == 0
    assert continuation.decisions == 0

    assert wrapper.decide(0, None, _events(3), menu) == menu[-1]["event"]
    assert continuation.decisions == 1
    assert continuation.player_ids == [0]
    assert continuation.started == [0]


def test_replay_preserves_omitted_passes_for_live_seats_until_global_cut() -> None:
    continuation = FakeContinuation()
    wrapper = ReplayThenEngine(
        "seat0",
        [],
        continuation,
        replay_until=(0, 0, 4),
        scripted_policy=True,
    )
    state = SimpleNamespace(last_cans=SimpleNamespace(can_discard=False))
    reaction = _menu("pass", "pon 3m", kinds=("none", "pon"))

    assert wrapper.auto_reaction(state, reaction, _events(2), 0) == reaction[0]["event"]
    assert continuation.auto_reactions == 0
    assert wrapper.auto_reaction(state, reaction, _events(5), 0) == reaction[-1]["event"]
    assert continuation.auto_reactions == 1


def test_fixed_controls_are_recomputed_even_before_the_replay_cut() -> None:
    continuation = FakeContinuation()
    wrapper = ReplayThenEngine(
        "seat1",
        [],
        continuation,
        replay_until=(0, 0, 4),
        scripted_policy=False,
    )
    state = SimpleNamespace(last_cans=SimpleNamespace(can_discard=False))
    reaction = _menu("pass", "pon 3m", kinds=("none", "pon"))
    assert wrapper.auto_reaction(state, reaction, _events(2), 0) == reaction[-1]["event"]
    assert continuation.auto_reactions == 1


def test_unrecorded_live_decision_before_cut_is_a_hard_error() -> None:
    wrapper = ReplayThenEngine(
        "seat0",
        [],
        FakeContinuation(),
        replay_until=(0, 0, 4),
        scripted_policy=True,
    )
    with pytest.raises(ReplayDiverged, match="unrecorded model decision"):
        wrapper.decide(
            0,
            None,
            _events(2),
            _menu("discard 1m", "discard 2m"),
        )


def test_replay_mismatch_is_a_hard_error() -> None:
    wrapper = ReplayThenEngine("seat0", [_decision()], FakeContinuation())
    with pytest.raises(ReplayDiverged, match="recorded"):
        wrapper.decide(
            0,
            None,
            _events(2),
            _menu("discard 1m", "discard 3m"),
        )
    with pytest.raises(ReplayDiverged, match="recorded at"):
        wrapper.decide(
            0,
            None,
            _events(3),
            _menu("discard 1m", "discard 2m"),
        )


def test_factual_kyoku_delta_sums_multi_ron() -> None:
    events = [
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
        },
        {"type": "hora", "deltas": [-3900, 3900, 0, 0]},
        {"type": "hora", "deltas": [-7700, 0, 7700, 0]},
        {"type": "end_kyoku"},
    ]
    assert factual_kyoku_point_delta(
        events, player_id=0, kyoku=0, honba=0
    ) == pytest.approx(-11600)


def _capsule_for_matching():
    from jongbench.experiments.capsule import ReplayCapsule

    decision = ScriptedDecision(
        sequence=0,
        seat="seat0",
        player_id=2,
        kyoku=0,
        honba=0,
        junme=1,
        tiles_left=69,
        event_count=2,
        menu=("discard 1m", "discard 2m"),
        choice=1,
        choice_label="discard 2m",
    )
    return ReplayCapsule(
        seed=(20260000, 1),
        rotation=2,
        table=("seat2", "seat3", "seat0", "seat1"),
        models=(
            ("seat0", "model/a"),
            ("seat1", "mortal"),
            ("seat2", "mortal"),
            ("seat3", "mortal"),
        ),
        evaluated_seat="seat0",
        profile="profile-a",
        control_checkpoint_sha256="a" * 64,
        control_use_policy=False,
        control_boltzmann_epsilon=0.0,
        control_boltzmann_temp=1.0,
        final_scores=(
            ("seat0", 30000),
            ("seat1", 20000),
            ("seat2", 26000),
            ("seat3", 24000),
        ),
        final_placements=(
            ("seat0", 1),
            ("seat1", 4),
            ("seat2", 2),
            ("seat3", 3),
        ),
        decisions=(decision,),
    )


def test_cached_baseline_is_joined_by_physical_chair() -> None:
    from jongbench.experiments.matched_control import match_control
    from jongbench.experiments.observations import AllControlBaseline

    capsule = _capsule_for_matching()
    checkpoint = "a" * 64
    baseline = AllControlBaseline.create(
        wall_id=capsule.wall_id,
        control=capsule.control_identity_for(checkpoint),
        scores=(22000.0, 23000.0, 27000.0, 28000.0),
        placements=(4, 3, 2, 1),
    )
    result = match_control(capsule, baseline)
    assert result.table_position == 2
    assert result.observed_score == 30000
    assert result.control_score == 27000
    assert result.score_delta == 3000
    assert result.control_placement == 2
    assert result.baseline_record_id == baseline.record_id


def test_matched_control_rejects_a_different_control_policy() -> None:
    from jongbench.experiments.matched_control import match_control
    from jongbench.experiments.observations import AllControlBaseline

    capsule = _capsule_for_matching()
    baseline = AllControlBaseline.create(
        wall_id=capsule.wall_id,
        control=capsule.control_identity_for("b" * 64),
        scores=(25000.0, 25000.0, 25000.0, 25000.0),
        placements=(1, 2, 3, 4),
    )
    with pytest.raises(ValueError, match="control (checkpoint|policy)"):
        match_control(capsule, baseline)


def test_matched_control_rejects_more_than_one_live_policy() -> None:
    from dataclasses import replace

    from jongbench.experiments.matched_control import match_control
    from jongbench.experiments.observations import AllControlBaseline

    capsule = replace(
        _capsule_for_matching(),
        models=(
            ("seat0", "model/a"),
            ("seat1", "model/b"),
            ("seat2", "mortal"),
            ("seat3", "mortal"),
        ),
    )
    baseline = AllControlBaseline.create(
        wall_id=capsule.wall_id,
        control=capsule.control_identity_for("a" * 64),
        scores=(25000.0, 25000.0, 25000.0, 25000.0),
        placements=(1, 2, 3, 4),
    )
    with pytest.raises(ValueError, match="exactly one live policy seat"):
        match_control(capsule, baseline)
