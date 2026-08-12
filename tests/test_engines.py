from __future__ import annotations

import copy
import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from types import SimpleNamespace

import jongbench
import libriichi
from jongbench import engines as engines_module
from jongbench import prompts, providers
from jongbench.engines import LLMEngine, RandomEngine, sanitize_events

_CONTEXT = threading.local()


class FakeProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 1200,
        temperature: float | None = None,
    ) -> providers.Completion:
        del max_tokens, temperature
        content = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        # The newest turn carries a cache breakpoint, so its content is a block list.
        # Each turn is checked as it is sent, so checking the newest covers every turn.
        user = (
            content
            if isinstance(content, str)
            else "".join(block["text"] for block in content)
        )
        with self._lock:
            self.calls.append(
                {
                    "player_id": getattr(_CONTEXT, "player_id", None),
                    "events_len": getattr(_CONTEXT, "events_len", None),
                    "start_key": getattr(_CONTEXT, "start_key", None),
                    "prompt": user,
                }
            )
        return providers.Completion(
            text='{"choice": 0}',
            reasoning="considered the wall",
            usage={"input_tokens": 10, "output_tokens": 5},
        )


class RecordingLLMEngine(LLMEngine):
    def __init__(
        self,
        name: str,
        fake_provider: FakeProvider,
        event_records: list[tuple[int, list[dict[str, Any]]]],
        decision_log: list[dict[str, Any]],
    ) -> None:
        super().__init__(
            name,
            "openai/fake",
            decision_log=decision_log,
            concurrency=1,
        )
        self.provider = fake_provider
        self.event_records = event_records

    def decide(
        self,
        player_id: int,
        state: Any,
        events: list[dict[str, Any]],
        menu: list[dict[str, Any]],
        game_index: int = 0,
    ) -> dict[str, Any]:
        copied = copy.deepcopy(events)
        self.event_records.append((player_id, copied))
        _CONTEXT.player_id = player_id
        _CONTEXT.events_len = len(events)
        _CONTEXT.start_key = _start_key(events)
        try:
            return super().decide(player_id, state, events, menu, game_index=game_index)
        finally:
            for attr in ("player_id", "events_len", "start_key"):
                try:
                    delattr(_CONTEXT, attr)
                except AttributeError:
                    pass


class RawCaptureSpectator:
    def __init__(self) -> None:
        self.records: list[tuple[int, int, list[dict[str, Any]]]] = []
        self._lock = threading.Lock()

    def publish(
        self,
        game_index: int,
        player_id: int,
        raw_events: list[dict[str, Any]],
    ) -> None:
        with self._lock:
            self.records.append((game_index, player_id, copy.deepcopy(raw_events)))


def test_sanitize_events() -> None:
    events = [
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "dora_marker": "3p",
            "scores": [25000, 25000, 25000, 25000],
            "tehais": [
                [
                    "1m",
                    "2m",
                    "3m",
                    "4m",
                    "5m",
                    "6m",
                    "7m",
                    "8m",
                    "9m",
                    "1p",
                    "2p",
                    "3p",
                    "4p",
                ],
                [
                    "5p",
                    "6p",
                    "7p",
                    "8p",
                    "9p",
                    "1s",
                    "2s",
                    "3s",
                    "4s",
                    "5s",
                    "6s",
                    "7s",
                    "8s",
                ],
                ["9s", "E", "E", "S", "S", "W", "W", "N", "N", "P", "P", "F", "F"],
                [
                    "C",
                    "C",
                    "1m",
                    "2m",
                    "3m",
                    "4p",
                    "5p",
                    "6p",
                    "7s",
                    "8s",
                    "9s",
                    "E",
                    "S",
                ],
            ],
        },
        {"type": "tsumo", "actor": 1, "pai": "5pr"},
        {"type": "tsumo", "actor": 2, "pai": "9s"},
        {"type": "dahai", "actor": 2, "pai": "9s", "tsumogiri": True},
    ]
    original = copy.deepcopy(events)

    sanitized = sanitize_events(events, 1)

    assert events == original
    assert sanitized is not events
    assert sanitized[0]["tehais"][1] == original[0]["tehais"][1]
    for seat in (0, 2, 3):
        assert sanitized[0]["tehais"][seat] == ["?"] * 13
    assert sanitized[1]["pai"] == "5pr"
    assert sanitized[2]["pai"] == "?"
    assert sanitized[3] == original[3]


def test_full_game_no_hidden_leak() -> None:
    fake = FakeProvider()
    spectator = RawCaptureSpectator()
    event_records: list[tuple[int, list[dict[str, Any]]]] = []
    decision_logs = [[], []]
    engines = [
        RecordingLLMEngine("fake-llm-0", fake, event_records, decision_logs[0]),
        RandomEngine("random-1", seed=101),
        RecordingLLMEngine("fake-llm-2", fake, event_records, decision_logs[1]),
        RandomEngine("random-3", seed=303),
    ]
    engines[0].spectator = spectator
    engines[2].spectator = spectator

    arena = libriichi.arena.FourEngines(disable_progress_bar=True)
    results = arena.py_4p(engines, (777, 3), 1)

    assert len(results) == 1, results
    names, scores, seed = results[0]
    del names, seed
    assert len(scores) == 4
    assert sum(scores) == 100000, scores

    assert event_records
    for player_id, events in event_records:
        for event in events:
            if event.get("type") == "start_kyoku":
                for seat, hand in enumerate(event["tehais"]):
                    if seat == player_id:
                        assert hand != ["?"] * 13
                    else:
                        assert hand == ["?"] * 13
            if event.get("type") == "tsumo" and event.get("actor") != player_id:
                assert event.get("pai") == "?"

    assert len(fake.calls) >= 10, len(fake.calls)
    finalized_logs = [
        raw_events
        for _, _, raw_events in spectator.records
        if raw_events and raw_events[-1].get("type") == "end_kyoku"
    ]
    assert finalized_logs
    hora_events = [
        event
        for raw_events in finalized_logs
        for event in raw_events
        if event.get("type") == "hora"
    ]
    assert hora_events
    assert all(isinstance(event.get("points"), int) for event in hora_events)
    assert all(event.get("yaku") for event in hora_events)
    total_records = sum(len(log) for log in decision_logs)
    assert total_records == len(fake.calls), (total_records, len(fake.calls))
    for engine, log in ((engines[0], decision_logs[0]), (engines[2], decision_logs[1])):
        assert engine.totals["calls"] == len(log)
        assert engine.totals["input_tokens"] == len(log) * 10
        assert engine.totals["output_tokens"] == len(log) * 5
        assert engine.totals["fallbacks"] == 0
        assert engine.totals["retries"] == 0
        assert all(
            record["usage"]
            == {
                "input_tokens": 10,
                "output_tokens": 5,
                "cached_input_tokens": 0,
                "reasoning_tokens": 0,
            }
            for record in log
        )
        assert all(record["raw_reasoning"] == "considered the wall" for record in log)
        assert all(record["prompt_version"] == 3 for record in log)
        assert all(record["state_hints"] is True for record in log)
        assert all(
            record["choice_label"] == record["menu"][record["choice"]] for record in log
        )

    raw_by_key = {
        (player_id, _start_key(raw_events), len(raw_events)): raw_events
        for _, player_id, raw_events in spectator.records
    }
    for call in fake.calls:
        key = (call["player_id"], call["start_key"], call["events_len"])
        raw_events = raw_by_key[key]
        start = _start_kyoku(raw_events)
        prompt = call["prompt"]
        assert "Engine-derived state hints" in prompt
        for seat, hand in enumerate(start["tehais"]):
            if seat == call["player_id"]:
                continue
            hidden_seq = "".join(str(tile) for tile in hand[:3])
            assert hidden_seq not in prompt, (call["player_id"], seat, hidden_seq)


def test_reasoning_token_caps_only_stop_runaways() -> None:
    assert LLMEngine("plain", "openai/fake").max_tokens == 4096
    assert LLMEngine("low", "openai/fake", reasoning="low").max_tokens == 64000
    assert LLMEngine("max", "openai/fake", reasoning="max").max_tokens == 96000
    assert (
        LLMEngine(
            "explicit", "openai/fake", reasoning="low", max_tokens=100000
        ).max_tokens
        == 100000
    )


def test_decision_coordinates_use_the_latest_kyoku() -> None:
    events = [
        {"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0},
        {"type": "tsumo", "actor": 2},
        {"type": "pon", "actor": 2},
        {"type": "end_kyoku"},
        {"type": "start_kyoku", "bakaze": "E", "kyoku": 2, "honba": 1},
        {"type": "tsumo", "actor": 0},
        {"type": "tsumo", "actor": 2},
        {"type": "chi", "actor": 2},
    ]

    assert engines_module._kyoku_id(events) == ("E", 2, 1)
    assert engines_module._decision_coords(2, events) == {
        "kyoku": 1,
        "honba": 1,
        "junme": 2,
        "tiles_left": 68,
    }


def test_random_engine_is_reproducible_across_batched_games() -> None:
    def play() -> list[Any]:
        engines = [RandomEngine(f"random-{seat}", seed=100 + seat) for seat in range(4)]
        arena = libriichi.arena.FourEngines(disable_progress_bar=True)
        return arena.py_4p(engines, (8181, 4), 4)

    assert play() == play()


def _start_kyoku(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in events:
        if event.get("type") == "start_kyoku":
            return event
    raise AssertionError("missing start_kyoku")


def _start_key(events: list[dict[str, Any]]) -> tuple[Any, ...]:
    start = _start_kyoku(events)
    return (
        start.get("bakaze"),
        start.get("kyoku"),
        start.get("honba"),
        start.get("kyotaku"),
        start.get("oya"),
        tuple(start.get("scores", [])),
        start.get("dora_marker"),
    )


def main() -> None:
    test_sanitize_events()
    test_full_game_no_hidden_leak()
    test_random_engine_is_reproducible_across_batched_games()
    print("OK")


if __name__ == "__main__":
    main()


def _engine_with_conversation(calls_enabled: bool, kyoku=("E", 1, 0)):
    engine = LLMEngine("toggle", "openai/fake", decision_log=[], concurrency=1)
    engine.provider = FakeProvider()
    history = engines_module._Conversation(kyoku=kyoku)
    history.calls_enabled = calls_enabled
    engine._conversations[0] = history
    return engine


_REACTION_EVENTS = [{"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0}]


def _state(can_discard: bool):
    return SimpleNamespace(last_cans=SimpleNamespace(can_discard=can_discard))


def test_extract_call_policy() -> None:
    assert prompts.extract_call_policy('{"choice": 2, "calls": "off"}') is False
    assert prompts.extract_call_policy('{"choice": 2, "calls":"on"}') is True
    assert prompts.extract_call_policy('{"choice": 2, "calls": "OFF"}') is False
    assert prompts.extract_call_policy('{"choice": 2}') is None
    # A reply naming both settings is ambiguous whichever end you read from.
    assert prompts.extract_call_policy('{"calls":"on"} ... {"calls":"off"}') is None


def test_declining_calls_skips_only_pure_call_reactions() -> None:
    engine = _engine_with_conversation(calls_enabled=False)
    menu = [
        {"kind": "none", "label": "pass", "event": {"type": "none"}},
        {"kind": "chi", "label": "chi", "event": {"type": "chi"}},
    ]
    assert engine.auto_reaction(_state(False), menu, _REACTION_EVENTS, 0) == {
        "type": "none"
    }
    assert engine.totals["calls_declined"] == 1


def test_declining_calls_never_suppresses_a_win() -> None:
    engine = _engine_with_conversation(calls_enabled=False)
    menu = [
        {"kind": "none", "label": "pass", "event": {"type": "none"}},
        {"kind": "hora", "label": "ron", "event": {"type": "hora"}},
    ]
    assert engine.auto_reaction(_state(False), menu, _REACTION_EVENTS, 0) is None
    assert engine.totals["calls_declined"] == 0


def test_declining_calls_never_touches_your_own_turn() -> None:
    engine = _engine_with_conversation(calls_enabled=False)
    menu = [
        {"kind": "discard", "label": "1m", "event": {"type": "dahai"}},
        {"kind": "ankan", "label": "ankan", "event": {"type": "ankan"}},
    ]
    assert engine.auto_reaction(_state(True), menu, _REACTION_EVENTS, 0) is None


def test_calls_are_accepted_again_in_a_new_kyoku() -> None:
    engine = _engine_with_conversation(calls_enabled=False, kyoku=("E", 1, 0))
    menu = [
        {"kind": "none", "label": "pass", "event": {"type": "none"}},
        {"kind": "pon", "label": "pon", "event": {"type": "pon"}},
    ]
    later = [{"type": "start_kyoku", "bakaze": "E", "kyoku": 2, "honba": 0}]
    assert engine.auto_reaction(_state(False), menu, later, 0) is None


def test_accepting_calls_asks_the_model() -> None:
    engine = _engine_with_conversation(calls_enabled=True)
    menu = [
        {"kind": "none", "label": "pass", "event": {"type": "none"}},
        {"kind": "pon", "label": "pon", "event": {"type": "pon"}},
    ]
    assert engine.auto_reaction(_state(False), menu, _REACTION_EVENTS, 0) is None


def test_auto_pass_reactions_skips_calls_without_any_conversation() -> None:
    """The engine-wide cost mode passes pure reactions even where the furo toggle
    would not fire: no conversation opened, calls never declined by the model."""
    engine = LLMEngine(
        "cheap", "openai/fake", decision_log=[], concurrency=1, auto_pass_reactions=True
    )
    engine.provider = FakeProvider()
    menu = [
        {"kind": "none", "label": "pass", "event": {"type": "none"}},
        {"kind": "chi", "label": "chi", "event": {"type": "chi"}},
    ]
    assert engine.auto_reaction(_state(False), menu, _REACTION_EVENTS, 0) == {
        "type": "none"
    }
    assert engine.totals["calls_declined"] == 1

    win = [
        {"kind": "none", "label": "pass", "event": {"type": "none"}},
        {"kind": "hora", "label": "ron", "event": {"type": "hora"}},
    ]
    assert engine.auto_reaction(_state(False), win, _REACTION_EVENTS, 0) is None
    own_turn = [
        {"kind": "discard", "label": "1m", "event": {"type": "dahai"}},
        {"kind": "pon", "label": "pon", "event": {"type": "pon"}},
    ]
    assert engine.auto_reaction(_state(True), own_turn, _REACTION_EVENTS, 0) is None


def test_make_engine_mortal_loads_the_checkpoint(monkeypatch) -> None:
    from jongbench import evaluate, positions

    loaded: list[str] = []
    monkeypatch.setattr(
        evaluate, "load_engine", lambda weights: loaded.append(weights) or object()
    )
    engine = engines_module.make_engine(
        "seat3", "mortal", weights="w.pth", decision_log=[], temperature=0.7
    )
    assert isinstance(engine, positions.MortalArenaEngine)
    assert engine.name == "seat3"
    assert loaded == ["w.pth"]
