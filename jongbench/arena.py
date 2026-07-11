from dataclasses import dataclass

import jongbench  # noqa: F401  (sets up the libriichi import path)
import libriichi


@dataclass
class GameSummary:
    seed: tuple[int, int]
    names: list[str]      # seat order at game start (East first)
    scores: list[int]     # same order
    placements: dict[str, int]  # name -> 1..4, ties broken by seat


def run_games(
    engines: list,
    games: int,
    seed_start: tuple[int, int] = (10000, 1),
    log_dir: str | None = None,
    disable_progress_bar: bool = True,
) -> list[GameSummary]:
    if len(engines) != 4:
        raise ValueError(f"expected exactly 4 engines, got {len(engines)}")
    if games <= 0:
        raise ValueError("games must be greater than zero")
    arena = libriichi.arena.FourEngines(
        disable_progress_bar=disable_progress_bar, log_dir=log_dir
    )
    results = arena.py_4p(engines, seed_start, games)

    summaries = []
    for names, scores, seed in results:
        order = sorted(range(4), key=lambda s: (-scores[s], s))
        placements = {names[s]: rank + 1 for rank, s in enumerate(order)}
        summaries.append(
            GameSummary(
                seed=seed,
                names=list(names),
                scores=list(scores),
                placements=placements,
            )
        )
    return summaries
