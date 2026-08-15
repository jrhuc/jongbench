from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jongbench
import libriichi
from jongbench.actions import build_menu


class MenuRandomEngine:
    engine_type = "mjai-log"

    def __init__(self, name: str, seed: int) -> None:
        self.name = name
        self._random = random.Random(seed)
        self.player_ids: list[int] | None = None
        self.menu_invocations = 0

    def set_player_ids(self, player_ids: list[int]) -> None:
        self.player_ids = player_ids

    def react_batch(self, game_states: list[Any]) -> list[str]:
        choices: list[str] = []
        for game_state in game_states:
            state = game_state.state
            menu = build_menu(state)
            self.menu_invocations += 1
            assert menu, state.brief_info()

            for item in menu:
                state.validate_reaction(
                    json.dumps(item["event"], separators=(",", ":"))
                )

            hora = [item for item in menu if item["kind"] == "hora"]
            if hora:
                chosen = hora[0]
            else:
                weighted = list(menu)
                passes = [item for item in menu if item["event"].get("type") == "none"]
                if passes:
                    weighted.extend([passes[0], passes[0]])
                chosen = self._random.choice(weighted)

            choices.append(json.dumps(chosen["event"], separators=(",", ":")))
        return choices

    def start_game(self, game_idx: int) -> None:
        pass

    def end_kyoku(self, game_idx: int) -> None:
        pass

    def end_game(self, game_idx: int, scores: list[int]) -> None:
        pass


def test_menu_validity() -> None:
    engines = [MenuRandomEngine(f"menu-random-{i}", 10_000 + i) for i in range(4)]
    with tempfile.TemporaryDirectory(prefix="jongbench-actions-") as log_dir:
        arena = libriichi.arena.FourEngines(disable_progress_bar=True, log_dir=log_dir)
        results = arena.py_4p(engines, (5000, 1), 3)

    assert len(results) == 3, results
    for names, scores, seed in results:
        assert len(names) == 4
        assert len(scores) == 4
        assert sum(scores) == 100000, (seed, scores)

    invocations = sum(engine.menu_invocations for engine in engines)
    assert invocations > 500, invocations

    print(f"games={len(results)} menu_invocations={invocations}")
    for names, scores, seed in results:
        print(f"seed={seed} scores={list(scores)} names={list(names)}")
    print("OK")


if __name__ == "__main__":
    test_menu_validity()
