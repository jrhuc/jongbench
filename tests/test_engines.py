from __future__ import annotations

import copy
import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jongbench
import libriichi
from jongbench.engines import LLMEngine, RandomEngine, sanitize_events


_CONTEXT = threading.local()


class FakeProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1200,
        temperature: float = 0.6,
    ) -> tuple[str, dict[str, int]]:
        del system, max_tokens, temperature
        with self._lock:
            self.calls.append(
                {
                    "player_id": getattr(_CONTEXT, "player_id", None),
                    "events_len": getattr(_CONTEXT, "events_len", None),
                    "start_key": getattr(_CONTEXT, "start_key", None),
                    "prompt": user,
                }
            )
        return '{"choice": 0}', {"input_tokens": 10, "output_tokens": 5}


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
            "openai:fake",
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
    ) -> dict[str, Any]:
        copied = copy.deepcopy(events)
        self.event_records.append((player_id, copied))
        _CONTEXT.player_id = player_id
        _CONTEXT.events_len = len(events)
        _CONTEXT.start_key = _start_key(events)
        try:
            return super().decide(player_id, state, events, menu)
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
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
                ["5p", "6p", "7p", "8p", "9p", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s"],
                ["9s", "E", "E", "S", "S", "W", "W", "N", "N", "P", "P", "F", "F"],
                ["C", "C", "1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "S"],
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
    total_records = sum(len(log) for log in decision_logs)
    assert total_records == len(fake.calls), (total_records, len(fake.calls))
    for engine, log in ((engines[0], decision_logs[0]), (engines[2], decision_logs[1])):
        assert engine.totals["calls"] == len(log)
        assert engine.totals["input_tokens"] == len(log) * 10
        assert engine.totals["output_tokens"] == len(log) * 5
        assert engine.totals["fallbacks"] == 0
        assert engine.totals["retries"] == 0
        assert all(record["usage"] == {"input_tokens": 10, "output_tokens": 5} for record in log)

    raw_by_key = {
        (player_id, _start_key(raw_events), len(raw_events)): raw_events
        for _, player_id, raw_events in spectator.records
    }
    for call in fake.calls:
        key = (call["player_id"], call["start_key"], call["events_len"])
        raw_events = raw_by_key[key]
        start = _start_kyoku(raw_events)
        prompt = call["prompt"]
        for seat, hand in enumerate(start["tehais"]):
            if seat == call["player_id"]:
                continue
            hidden_seq = "".join(str(tile) for tile in hand[:3])
            assert hidden_seq not in prompt, (call["player_id"], seat, hidden_seq)


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
