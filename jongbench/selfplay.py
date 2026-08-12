from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import jongbench  # noqa: F401
import libriichi
from jongbench.arena import GameSummary, run_games
from jongbench.evaluate import load_engine, resolve_device
from jongbench.mortal_engine import MortalEngine


def _share_engine(template: MortalEngine, name: str, **overrides: Any) -> MortalEngine:
    kwargs = {
        "is_oracle": template.is_oracle,
        "version": template.version,
        "device": template.device,
        "enable_amp": template.enable_amp,
        "enable_quick_eval": template.enable_quick_eval,
        "enable_rule_based_agari_guard": template.enable_rule_based_agari_guard,
        "name": name,
        "boltzmann_epsilon": template.boltzmann_epsilon,
        "boltzmann_temp": template.boltzmann_temp,
        "top_p": template.top_p,
        "policy": template.policy,
        "use_policy": template.use_policy,
        "aux_net": template.aux_net,
        "confidence": template.confidence,
    }
    kwargs.update(overrides)
    return MortalEngine(template.brain, template.dqn, **kwargs)


def make_play_engines(
    weights: str,
    *,
    n: int = 4,
    device: str | torch.device = "auto",
    use_policy: bool = False,
    boltzmann_epsilon: float = 0.0,
    boltzmann_temp: float = 0.2,
    name_prefix: str = "mortal",
    enable_quick_eval: bool = True,
) -> list[MortalEngine]:
    device_t = resolve_device(device)
    first = load_engine(
        weights,
        device=device_t,
        use_policy=use_policy,
        enable_quick_eval=enable_quick_eval,
        enable_amp=device_t.type == "cuda",
        boltzmann_epsilon=boltzmann_epsilon,
        boltzmann_temp=boltzmann_temp,
        name=f"{name_prefix}-0",
    )
    engines = [first]
    for i in range(1, n):
        engines.append(_share_engine(first, f"{name_prefix}-{i}"))
    return engines


def selfplay(
    *,
    weights: str,
    out_dir: str | Path,
    games: int,
    seed: int = 10000,
    key: int = 1,
    batch_games: int = 32,
    device: str | torch.device = "auto",
    use_policy: bool = False,
    boltzmann_epsilon: float = 0.0,
    boltzmann_temp: float = 0.2,
    disable_progress_bar: bool = False,
) -> list[GameSummary]:
    if games <= 0:
        raise ValueError("games must be greater than zero")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    engines = make_play_engines(
        weights,
        device=device,
        use_policy=use_policy,
        boltzmann_epsilon=boltzmann_epsilon,
        boltzmann_temp=boltzmann_temp,
    )
    summaries: list[GameSummary] = []
    remaining = games
    seed_cursor = seed
    while remaining:
        n = min(batch_games, remaining)
        print(f"selfplay {n} games seed={seed_cursor} remaining={remaining}")
        summaries.extend(
            run_games(
                engines,
                n,
                seed_start=(seed_cursor, key),
                log_dir=str(out),
                disable_progress_bar=disable_progress_bar,
            )
        )
        seed_cursor += n
        remaining -= n
    return summaries


@dataclass
class DuelResult:
    rankings: list[int]
    games: int
    avg_rank: float
    avg_pt: float
    pts: tuple[float, float, float, float] = (90.0, 45.0, 0.0, -135.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rankings": self.rankings,
            "games": self.games,
            "avg_rank": self.avg_rank,
            "avg_pt": self.avg_pt,
            "pts": list(self.pts),
        }


def duel(
    *,
    challenger_weights: str,
    champion_weights: str,
    games: int,
    seed: int = 20000,
    key: int = 1,
    device: str | torch.device = "auto",
    challenger_policy: bool = False,
    champion_policy: bool = False,
    log_dir: str | Path | None = None,
    disable_progress_bar: bool = False,
    pts: tuple[float, float, float, float] = (90.0, 45.0, 0.0, -135.0),
) -> DuelResult:
    if games <= 0:
        raise ValueError("games must be greater than zero")
    if games % 4 != 0:
        raise ValueError(
            "duel games must be a multiple of 4 (one seed covers four seats)"
        )
    device_t = resolve_device(device)
    challenger = load_engine(
        challenger_weights,
        device=device_t,
        use_policy=challenger_policy,
        enable_quick_eval=True,
        enable_amp=device_t.type == "cuda",
        name="challenger",
    )
    if Path(champion_weights).resolve() == Path(challenger_weights).resolve() and not (
        challenger_policy or champion_policy
    ):
        champion = _share_engine(challenger, "champion", use_policy=False)
    else:
        champion = load_engine(
            champion_weights,
            device=device_t,
            use_policy=champion_policy,
            enable_quick_eval=True,
            enable_amp=device_t.type == "cuda",
            name="champion",
        )
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    seed_count = games // 4
    arena = libriichi.arena.OneVsThree(
        disable_progress_bar=disable_progress_bar,
        log_dir=None if log_dir is None else str(log_dir),
    )
    rankings = list(arena.py_vs_py(challenger, champion, (seed, key), seed_count))
    total = sum(rankings)
    avg_rank = sum((i + 1) * n for i, n in enumerate(rankings)) / max(total, 1)
    avg_pt = sum(pts[i] * n for i, n in enumerate(rankings)) / max(total, 1)
    return DuelResult(
        rankings=rankings,
        games=total,
        avg_rank=avg_rank,
        avg_pt=avg_pt,
        pts=pts,
    )
