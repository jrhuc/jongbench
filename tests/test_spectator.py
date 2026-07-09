from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jongbench  # noqa: F401
import libriichi
from jongbench.spectator import Spectator, TableState, render_table


class TsumogiriEngine:
    engine_type = "mjai-log"

    def __init__(self, name: str) -> None:
        self.name = name
        self.player_ids: list[int] | None = None

    def set_player_ids(self, player_ids: list[int]) -> None:
        self.player_ids = player_ids

    def react_batch(self, game_states: list[Any]) -> list[str]:
        assert self.player_ids is not None
        reactions = []
        for game_state in game_states:
            player_id = self.player_ids[game_state.game_index]
            state = game_state.state
            if state.last_cans.can_discard:
                reactions.append(
                    json.dumps(
                        {
                            "type": "dahai",
                            "actor": player_id,
                            "pai": state.last_self_tsumo(),
                            "tsumogiri": True,
                        },
                        separators=(",", ":"),
                    )
                )
            else:
                reactions.append('{"type":"none"}')
        return reactions

    def start_game(self, game_idx: int) -> None:
        pass

    def end_kyoku(self, game_idx: int) -> None:
        pass

    def end_game(self, game_idx: int, scores: list[int]) -> None:
        pass


def main() -> None:
    events = _generate_log()
    assert events

    table = TableState()
    seen_dahai = 0
    seen_discard_total = 0
    for idx, event in enumerate(events):
        before_tiles_left = table.tiles_left
        before_discards = sum(len(row) for row in table.discards)
        table.apply(event)
        assert len(table.scores) == 4
        assert all(isinstance(score, int) for score in table.scores)

        if event["type"] == "start_kyoku":
            assert table.tiles_left == 70
            seen_discard_total = 0
        elif event["type"] == "tsumo":
            assert table.tiles_left == before_tiles_left - 1, (idx, event, before_tiles_left, table.tiles_left)
        elif event["type"] == "dahai":
            seen_dahai += 1
            seen_discard_total += 1
            total_discards = sum(len(row) for row in table.discards)
            assert total_discards == before_discards + 1, (idx, event)
            assert total_discards == seen_discard_total, (idx, event, total_discards, seen_discard_total)
            for seat in range(4):
                tile_equiv = len(table.hands[seat]) + 3 * len(table.melds[seat])
                assert tile_equiv in {13, 14}, (idx, seat, tile_equiv, event, table.snapshot())

    assert seen_dahai > 0
    assert any(table.discards)

    kyokus = _split_kyokus(events)
    assert kyokus
    updates = 0

    def on_update(_: TableState) -> None:
        nonlocal updates
        updates += 1

    spectator = Spectator(on_update=on_update)
    expected = 0
    for kyoku_events in kyokus:
        n_events = len(kyoku_events)
        prefixes = sorted({1, min(2, n_events), min(5, n_events), max(1, n_events // 2), n_events})
        for idx, prefix in enumerate(prefixes):
            spectator.publish(0, idx % 4, kyoku_events[:prefix])
            spectator.publish(0, (idx + 1) % 4, kyoku_events[:prefix])
            assert len(spectator.events_since(0)) == expected + prefix
            assert updates == expected + prefix
        expected += n_events
        for seat in range(4):
            spectator.publish(0, seat, kyoku_events)
            assert len(spectator.events_since(0)) == expected
            assert updates == expected

    spectator.publish(1, 0, kyokus[0])
    assert len(spectator.events_since(0)) == expected
    assert updates == expected

    feed = spectator.events_since(0)
    assert [item["seq"] for item in feed] == list(range(1, len(feed) + 1))
    assert [item["event"] for item in feed] == [event for kyoku in kyokus for event in kyoku]

    first_table = TableState()
    if events[0]["type"] == "start_game":
        first_table.apply(events[0])
    for event in kyokus[0]:
        first_table.apply(event)
    print(render_table(first_table))
    print("OK")


def _generate_log() -> list[dict[str, Any]]:
    engines = [TsumogiriEngine(f"tsumogiri-{idx}") for idx in range(4)]
    with tempfile.TemporaryDirectory(prefix="jongbench-spectator-") as tempdir:
        arena = libriichi.arena.FourEngines(disable_progress_bar=True, log_dir=tempdir)
        arena.py_4p(engines, (321, 9), 1)
        logs = sorted(Path(tempdir).glob("*.json.gz"))
        assert len(logs) == 1
        with gzip.open(logs[0], "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]


def _split_kyokus(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    kyokus: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for event in events:
        if event["type"] == "start_kyoku":
            if current:
                kyokus.append(current)
            current = [event]
        elif current:
            current.append(event)
            if event["type"] == "end_kyoku":
                kyokus.append(current)
                current = []
    if current:
        kyokus.append(current)
    return kyokus


if __name__ == "__main__":
    main()
