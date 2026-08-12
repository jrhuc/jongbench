from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jongbench
import libriichi
from jongbench.evaluate import load_engine, load_mjai_log, review_game, review_player


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
                tile = state.last_self_tsumo()
                reactions.append(
                    json.dumps(
                        {
                            "type": "dahai",
                            "actor": player_id,
                            "pai": tile,
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


def test_mortal_review() -> None:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tempdir:
        arena = libriichi.arena.FourEngines(
            disable_progress_bar=True,
            log_dir=tempdir,
        )
        engines = [TsumogiriEngine(f"tsumogiri-{idx}") for idx in range(4)]
        arena.py_4p(engines, (9000, 5), 1)
        logs = sorted(Path(tempdir).glob("*.json.gz"))
        assert len(logs) == 1
        events = load_mjai_log(str(logs[0]))

    end_kyoku_idx = next(
        idx for idx, event in enumerate(events) if event["type"] == "end_kyoku"
    )
    events = events[: end_kyoku_idx + 1] + [{"type": "end_game"}]

    engine = load_engine("weights/mortal.pth")
    review = review_player(events, 0, engine)

    assert review["total_reviewed"] > 0
    assert 0.0 <= review["rating"] <= 1.0
    for entry in review["entries"]:
        prob_sum = sum(detail["prob"] for detail in entry["details"])
        assert abs(prob_sum - 1.0) <= 1e-3
        assert entry["actual_index"] < len(entry["details"])
    assert any(not entry["is_equal"] for entry in review["entries"])
    batched = review_game(events, engine)[0]
    assert batched["total_reviewed"] == review["total_reviewed"]
    assert batched["total_matches"] == review["total_matches"]
    assert batched["rating"] == pytest.approx(review["rating"], rel=1e-6)
    assert [entry["actual"] for entry in batched["entries"]] == [
        entry["actual"] for entry in review["entries"]
    ]

    elapsed = time.perf_counter() - started
    print(
        f"total_reviewed={review['total_reviewed']} "
        f"rating={review['rating']:.6f} "
        f"elapsed={elapsed:.3f}s"
    )
    print("OK")


if __name__ == "__main__":
    test_mortal_review()
