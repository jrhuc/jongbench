from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from jongbench.selfplay import DUPLICATE_PTS, DuelResult, duel
from jongbench.train import PolicyRLConfig, train_policy_rl


@dataclass
class ImproveConfig:
    init: str
    out_dir: str
    control: str | None = "weights/mortal.pth"
    rounds: int = 4
    rollout_games: int = 256
    updates: int = 128
    batch_size: int = 512
    duel_games: int = 512
    device: str = "auto"
    seed: int = 20270000
    lr: float = 1e-4
    rollout_temperature: float = 1.0
    clip_ratio: float = 0.2
    target_kl: float | None = 0.03
    anchor_kl_weight: float = 0.02
    entropy_weight: float = 0.001
    promotion_z: float = 1.0
    pts: tuple[float, float, float, float] = DUPLICATE_PTS
    promotion_margin: float = 0.0


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _promotion_score(result: DuelResult, z: float) -> float:
    if result.standard_error is None:
        return float("-inf")
    return result.avg_pt - z * result.standard_error


def improve_policy(cfg: ImproveConfig) -> dict[str, Any]:
    if cfg.rounds <= 0:
        raise ValueError("rounds must be positive")
    if cfg.rollout_games <= 0 or cfg.rollout_games % 4:
        raise ValueError("rollout_games must be a positive multiple of 4")
    if cfg.duel_games < 8 or cfg.duel_games % 4:
        raise ValueError("duel_games must be a multiple of 4 and at least 8")
    if cfg.promotion_z < 0:
        raise ValueError("promotion_z must be non-negative")
    initial = Path(cfg.init)
    if not initial.is_file():
        raise FileNotFoundError(f"initial checkpoint does not exist: {initial}")
    if cfg.control is not None and not Path(cfg.control).is_file():
        raise FileNotFoundError(f"control checkpoint does not exist: {cfg.control}")

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("round-*")) or (out / "league.json").exists():
        raise FileExistsError(f"league output already contains a run: {out}")

    champion = initial
    rounds: list[dict[str, Any]] = []
    for round_index in range(1, cfg.rounds + 1):
        round_dir = out / f"round-{round_index:02d}"
        logs_dir = round_dir / "logs"
        candidate = round_dir / "candidate.pth"
        duel_dir = round_dir / "duel"
        round_dir.mkdir(parents=True)
        rollout_seed = cfg.seed + (round_index - 1) * 1_000_000
        duel_seed = rollout_seed + cfg.rollout_games + 10_000

        print(f"round {round_index}/{cfg.rounds}: rollout from {champion}")
        rollout = duel(
            challenger_weights=str(champion),
            champion_weights=str(champion),
            games=cfg.rollout_games,
            seed=rollout_seed,
            device=cfg.device,
            challenger_policy=True,
            champion_policy=True,
            challenger_boltzmann_epsilon=1.0,
            challenger_boltzmann_temp=cfg.rollout_temperature,
            challenger_agari_guard=False,
            log_dir=logs_dir,
            disable_progress_bar=True,
            pts=cfg.pts,
        )
        training = train_policy_rl(
            PolicyRLConfig(
                logs=str(logs_dir),
                init=str(champion),
                anchor=str(initial),
                out=str(candidate),
                steps=cfg.updates,
                batch_size=cfg.batch_size,
                device=cfg.device,
                lr=cfg.lr,
                clip_ratio=cfg.clip_ratio,
                target_kl=cfg.target_kl,
                anchor_kl_weight=cfg.anchor_kl_weight,
                entropy_weight=cfg.entropy_weight,
                sampling_temperature=cfg.rollout_temperature,
                duplicate_challenger_only=True,
                pts=cfg.pts,
                seed=cfg.seed + round_index,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result = duel(
            challenger_weights=str(candidate),
            champion_weights=str(champion),
            games=cfg.duel_games,
            seed=duel_seed,
            device=cfg.device,
            challenger_policy=True,
            champion_policy=True,
            log_dir=duel_dir,
            disable_progress_bar=True,
            pts=cfg.pts,
        )
        promotion_score = _promotion_score(result, cfg.promotion_z)
        promoted = promotion_score > cfg.promotion_margin
        previous = champion
        if promoted:
            champion = candidate
        record = {
            "round": round_index,
            "rollout_seed": rollout_seed,
            "duel_seed": duel_seed,
            "source": str(previous),
            "candidate": str(candidate),
            "rollout": rollout.as_dict(),
            "training": training,
            "duel": result.as_dict(),
            "promotion_score": promotion_score,
            "promoted": promoted,
        }
        rounds.append(record)
        _write_json(round_dir / "round.json", record)
        verdict = "promoted" if promoted else "rejected"
        print(
            f"round {round_index}: {verdict} avg_pt={result.avg_pt:.3f} "
            f"se={result.standard_error:.3f} score={promotion_score:.3f}"
        )

    champion_out = out / "champion.pth"
    if champion.resolve() != champion_out.resolve():
        shutil.copy2(champion, champion_out)

    final_evaluations: dict[str, Any] = {}
    final_seed = cfg.seed + cfg.rounds * 1_000_000 + 100_000
    if champion.resolve() != initial.resolve():
        initial_result = duel(
            challenger_weights=str(champion_out),
            champion_weights=str(initial),
            games=cfg.duel_games,
            seed=final_seed,
            device=cfg.device,
            challenger_policy=True,
            champion_policy=True,
            log_dir=out / "final-vs-initial",
            disable_progress_bar=True,
            pts=cfg.pts,
        )
        final_evaluations["initial_policy"] = initial_result.as_dict()
        final_seed += cfg.duel_games // 4
    if cfg.control is not None:
        control_result = duel(
            challenger_weights=str(champion_out),
            champion_weights=cfg.control,
            games=cfg.duel_games,
            seed=final_seed,
            device=cfg.device,
            challenger_policy=True,
            champion_policy=False,
            log_dir=out / "final-vs-control",
            disable_progress_bar=True,
            pts=cfg.pts,
        )
        final_evaluations["control_q"] = control_result.as_dict()

    manifest = {
        "format": 1,
        "config": asdict(cfg),
        "initial": str(initial),
        "champion": str(champion_out),
        "promotions": sum(int(record["promoted"]) for record in rounds),
        "rounds": rounds,
        "final_evaluations": final_evaluations,
    }
    _write_json(out / "league.json", manifest)
    return manifest
