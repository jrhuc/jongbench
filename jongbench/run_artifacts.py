from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import providers
from .arena import GameSummary
from .artifacts import load_mjai_log

_RUN_DIRECTORIES = ("logs", "decisions", "review")


def safe_name(value: str, *, fallback: str = "player") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.+-]+", "_", value.strip()).strip("._-")
    return (cleaned or fallback)[:60]


def engine_names(specs: Sequence[str]) -> list[str]:
    """Return deterministic, filesystem-safe unique names for four engine specs."""
    seen: dict[str, int] = {}
    names: list[str] = []
    for seat, value in enumerate(specs):
        spec = providers.parse_spec(value)
        base = safe_name(spec.display_name or f"P{seat}")
        count = seen.get(base, 0) + 1
        seen[base] = count
        names.append(base if count == 1 else f"{base}-{count}")
    return names


def create_run_dir(root: str | Path, label: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = safe_name(label, fallback="run")
    base = Path(root) / f"{stamp}-{slug}"
    path = base
    suffix = 2
    while path.exists():
        path = Path(f"{base}-{suffix}")
        suffix += 1
    prepare_run_dir(path)
    return path


def prepare_run_dir(run_dir: str | Path) -> Path:
    path = Path(run_dir)
    for name in _RUN_DIRECTORIES:
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def run_log_paths(run_dir: str | Path) -> list[Path]:
    directory = Path(run_dir) / "logs"
    return sorted(
        [*directory.glob("*.json.gz"), *directory.glob("*.json")],
        key=log_sort_key,
    )


def write_run_config(
    run_dir: str | Path,
    *,
    label: str,
    models: Sequence[str],
    names: Sequence[str],
    games: int,
    seed_start: tuple[int, int],
    state_hints: bool,
    human_seat: int | None = None,
    no_eval: bool = False,
) -> dict[str, Any]:
    path = prepare_run_dir(run_dir)
    data: dict[str, Any] = {
        "label": label or path.name,
        "created": datetime.now(UTC).isoformat(),
        "models": list(models),
        "names": list(names),
        "games": int(games),
        "seed_start": [int(seed_start[0]), int(seed_start[1])],
        "state_hints": bool(state_hints),
        "human_seat": human_seat,
        "no_eval": bool(no_eval),
    }
    (path / "config.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return data


def _update_run_config(run_dir: str | Path, key: str, value: Any) -> None:
    path = Path(run_dir) / "config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data[key] = value
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def record_reviewer_checkpoint(run_dir: str | Path, checkpoint: Any) -> dict[str, Any]:
    """Attach the resolved grading checkpoint to an existing run config."""
    identity = (
        checkpoint.as_dict() if hasattr(checkpoint, "as_dict") else dict(checkpoint)
    )
    _update_run_config(run_dir, "reviewer_checkpoint", identity)
    return identity


def record_gameplay_checkpoints(
    run_dir: str | Path, engines: Sequence[Any]
) -> dict[str, dict[str, Any]]:
    """Record every checkpoint that directly selected gameplay actions."""
    identities = {
        str(engine.name): checkpoint.as_dict()
        for engine in engines
        if (checkpoint := getattr(engine, "checkpoint", None)) is not None
    }
    if identities:
        _update_run_config(run_dir, "gameplay_checkpoints", identities)
    return identities


def review_log(
    log_path: str | Path,
    summary: GameSummary,
    mortal: Any,
    *,
    temperature: float = 0.1,
    check_cancelled: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Review one game and construct the sole canonical review artifact payload."""
    from . import evaluate

    check = check_cancelled or (lambda: None)
    check()
    events = load_mjai_log(log_path)
    check()
    reviews = evaluate.review_game(
        events,
        mortal,
        temperature=temperature,
        check_cancelled=check,
    )
    check()
    players: dict[str, Any] = {}
    for seat in range(4):
        review = reviews[seat]
        players[str(seat)] = {
            "name": summary.names[seat],
            "review": review,
            "aggregates": evaluate.aggregates(review),
        }
    payload = {
        "seed": [int(summary.seed[0]), int(summary.seed[1])],
        "names": list(summary.names),
        "scores": list(summary.scores),
        "placements": dict(summary.placements),
        "players": players,
    }
    checkpoint = getattr(mortal, "checkpoint", None)
    if checkpoint is not None:
        payload["reviewer_checkpoint"] = checkpoint.as_dict()
    return payload


def write_review(run_dir: str | Path, payload: dict[str, Any]) -> Path:
    seed = payload.get("seed")
    if not isinstance(seed, list) or len(seed) < 2:
        raise ValueError("review payload seed must contain two integers")
    directory = Path(run_dir) / "review"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{int(seed[0])}_{int(seed[1])}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def summary_by_seed(
    summaries: Sequence[GameSummary] | Mapping[tuple[int, int], GameSummary] | None,
) -> dict[tuple[int, int], GameSummary]:
    if summaries is None:
        return {}
    if isinstance(summaries, Mapping):
        return {tuple(key): value for key, value in summaries.items()}
    return {tuple(summary.seed): summary for summary in summaries}


def reconstruct_summary(events: list[dict[str, Any]], path: Path) -> GameSummary:
    seed = seed_from_events_or_path(events, path)
    start_game = next(
        (event for event in events if event.get("type") == "start_game"), {}
    )
    names = [str(name) for name in start_game.get("names") or []]
    if len(names) != 4:
        names = [f"P{seat}" for seat in range(4)]

    last_scores = [25000, 25000, 25000, 25000]
    last_start_index = -1
    for index, event in enumerate(events):
        if event.get("type") != "start_kyoku":
            continue
        scores = event.get("scores")
        if isinstance(scores, list) and len(scores) == 4:
            last_scores = [int(score) for score in scores]
            last_start_index = index

    scores = list(last_scores)
    for event in events[last_start_index + 1 :]:
        event_type = event.get("type")
        if event_type == "reach_accepted":
            actor = event.get("actor")
            if isinstance(actor, int) and 0 <= actor < 4:
                scores[actor] -= 1000
            continue
        if event_type not in {"hora", "ryukyoku"}:
            continue
        deltas = event.get("deltas")
        if isinstance(deltas, list) and len(deltas) == 4:
            scores = [
                score + int(delta) for score, delta in zip(scores, deltas, strict=True)
            ]

    if any(event.get("type") == "end_game" for event in events):
        outstanding_kyotaku = 100000 - sum(scores)
        if outstanding_kyotaku > 0:
            leader = min(range(4), key=lambda seat: (-scores[seat], seat))
            scores[leader] += outstanding_kyotaku

    return GameSummary(
        seed=seed, names=names, scores=scores, placements=_placements(names, scores)
    )


def _placements(names: Sequence[str], scores: Sequence[int]) -> dict[str, int]:
    order = sorted(range(4), key=lambda seat: (-int(scores[seat]), seat))
    return {str(names[seat]): rank + 1 for rank, seat in enumerate(order)}


def reviews_missing(run_dir: Path) -> bool:
    logs = run_log_paths(run_dir)
    if not logs:
        raise ValueError(f"no logs found in {run_dir / 'logs'}")
    for log_path in logs:
        seed = seed_from_path(log_path)
        if not (run_dir / "review" / f"{seed_label(seed)}.json").exists():
            return True
    return False


def seed_from_events_or_path(
    events: list[dict[str, Any]],
    path: Path,
) -> tuple[int, int]:
    for event in events:
        if event.get("type") != "start_game":
            continue
        seed = event.get("seed")
        if isinstance(seed, list | tuple) and len(seed) >= 2:
            return int(seed[0]), int(seed[1])
    return seed_from_path(path)


def seed_from_path(path: Path) -> tuple[int, int]:
    name = path.name
    if name.endswith(".json.gz"):
        stem = name[:-8]
    elif name.endswith(".json"):
        stem = name[:-5]
    else:
        stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return int(parts[0]), int(parts[1])
    raise ValueError(f"cannot parse seed from {path.name}")


def seed_label(seed: tuple[int, int]) -> str:
    return f"{int(seed[0])}_{int(seed[1])}"


def log_sort_key(path: Path) -> tuple[int, int, str]:
    try:
        seed = seed_from_path(path)
    except ValueError:
        return 0, 0, path.name
    return seed[0], seed[1], path.name


def _select_log(run_dir: Path, game: str | None) -> Path:
    logs = run_log_paths(run_dir)
    if not logs:
        raise ValueError(f"no logs found in {run_dir / 'logs'}")
    if game is None:
        return logs[0]
    target = game.removesuffix(".json.gz").removesuffix(".json")
    for path in logs:
        if seed_label(seed_from_path(path)) == target:
            return path
    raise ValueError(f"game not found: {game}")


def build_replay_bundle(run_dir: Path, game: str | None = None) -> dict[str, Any]:
    """One game as the web replay viewer wants it: every mjai event paired with the
    table snapshot after it, plus standings and the Mortal review when present."""
    from .spectator import TableState

    log_path = _select_log(run_dir, game)
    events = load_mjai_log(log_path)
    seed = seed_from_events_or_path(events, log_path)
    review_path = run_dir / "review" / f"{seed_label(seed)}.json"
    review_data = (
        json.loads(review_path.read_text(encoding="utf-8"))
        if review_path.exists()
        else None
    )

    table = TableState()
    frames = []
    for seq, event in enumerate(events, start=1):
        table.apply(event)
        frames.append({"seq": seq, "event": event, "snapshot": table.snapshot()})

    if isinstance(review_data, dict):
        names = [str(name) for name in review_data.get("names") or []]
        scores = [int(score) for score in review_data.get("scores") or []]
        placements = dict(review_data.get("placements") or {})
    else:
        summary = reconstruct_summary(events, log_path)
        names, scores, placements = summary.names, summary.scores, summary.placements

    bundle: dict[str, Any] = {
        "game": seed_label(seed),
        "seed": [seed[0], seed[1]],
        "names": names,
        "scores": scores,
        "placements": placements,
        "frames": frames,
    }
    if isinstance(review_data, dict):
        bundle["review"] = review_data
    return bundle
