from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import decision_filename

_TILE_VIEWBOX = "0 0 320 446"
_VIEWBOX_RE = re.compile(r'\bviewBox\s*=\s*["\']([^"\']+)["\']')
_XML_RE = re.compile(r"^\s*<\?xml[^>]*>\s*", re.IGNORECASE)
_WEBUI_SPRITE_RE = re.compile(
    r'<div\s+id=["\']sprite["\']\s+hidden>(.*?)</div>',
    re.DOTALL,
)


def summarize(run_dir: str) -> dict[str, Any]:
    run = Path(run_dir)
    config = _read_json(run / "config.json", {})
    names = [str(name) for name in config.get("names") or []]
    specs = [str(spec) for spec in config.get("models") or []]
    accum: dict[str, dict[str, Any]] = {}

    for index, name in enumerate(names):
        accum[name] = _new_accumulator(name, specs[index] if index < len(specs) else "")

    games: list[dict[str, Any]] = []
    review_dir = run / "review"
    for path in (
        sorted(review_dir.glob("*.json"), key=_review_sort_key)
        if review_dir.exists()
        else []
    ):
        raw_game = _read_json(path, {})
        game = _game_row(path, raw_game)
        games.append(game)
        for player in game["players"]:
            name = player["name"]
            if name not in accum:
                accum[name] = _new_accumulator(name, "")
            _add_player(accum[name], player)

    decisions_dir = run / "decisions"
    engines = []
    for name, acc in accum.items():
        engine = _finalize_engine(acc)
        engine.update(_decision_stats(decisions_dir, name))
        engines.append(engine)

    engines.sort(
        key=lambda item: (
            -float(item["mean_rating"]),
            float(item["avg_placement"]) if item["games"] else 99.0,
            item["name"],
        )
    )
    for index, engine in enumerate(engines):
        engine["rank"] = index + 1
        engine["series_index"] = index

    summary = {
        "run_dir": str(run),
        "generated": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "engines": {engine["name"]: engine for engine in engines},
        "leaderboard": engines,
        "games": games,
        "review_files": len(games),
        "missing_review_files": max(0, int(config.get("games") or 0) - len(games)),
    }
    (run / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return summary


def write_report(run_dir: str, summary: dict[str, Any] | None = None) -> str:
    run = Path(run_dir)
    summary = summarize(run_dir) if summary is None else summary
    label = str(summary.get("config", {}).get("label") or run.name)
    sprite = _load_pai_sprite()
    view_box = _sprite_view_box(sprite)
    data = dict(summary)
    data["tile_view_box"] = view_box
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    report = _html_document(label, sprite, data_json)
    path = run / "report.html"
    path.write_text(report, encoding="utf-8")
    return str(path)


def leaderboard(batch_dir: str) -> dict[str, Any]:
    """Pool a batch of episode directories into one table keyed by model spec.

    A hanchan batch writes one run directory per episode, so a spec's placement
    distribution only exists across directories. Seat rotation moves a spec between
    engine names, which is why the pooling key is the spec and not the name."""
    root = Path(batch_dir)
    runs = (
        [root]
        if (root / "config.json").exists()
        else sorted(path.parent for path in root.glob("*/config.json"))
    )
    accum: dict[str, dict[str, Any]] = {}
    episodes: list[dict[str, Any]] = []

    for run in runs:
        summary = summarize(str(run))
        config = summary.get("config") or {}
        specs = {
            name: engine["spec"] or name for name, engine in summary["engines"].items()
        }

        for name, engine in summary["engines"].items():
            acc = accum.setdefault(specs[name], _new_batch_accumulator(specs[name]))
            acc["names"].add(name)
            # A spec can hold more than one seat in an episode, so count directories.
            acc["episodes"].add(str(run))
            records = int(engine.get("decision_records") or 0)
            acc["decision_records"] += records
            acc["fallback_count"] += int(engine.get("fallback_count") or 0)
            acc["input_tokens"] += int(engine.get("input_tokens") or 0)
            acc["output_tokens"] += int(engine.get("output_tokens") or 0)
            if engine.get("cost") is not None:
                acc["cost"] = (acc["cost"] or 0.0) + float(engine["cost"])
            latency = engine.get("mean_latency_ms")
            if latency is not None and records:
                acc["latency_sum"] += float(latency) * records
                acc["latency_records"] += records

        graded = bool(summary["games"])
        for row in summary["games"] or _final_players(config):
            for player in row["players"]:
                spec = specs.get(player["name"], player["name"])
                acc = accum.setdefault(spec, _new_batch_accumulator(spec))
                acc["names"].add(player["name"])
                acc["table_positions"].add(int(player["seat"]))
                _add_player(acc, player)
                if not graded:
                    # An ungraded episode has no rating; a zero would read as a bad one.
                    acc["ratings"].pop()
                placement = _int_or_none(player.get("placement"))
                if placement is not None and 1 <= placement <= 4:
                    acc["placement_counts"][placement - 1] += 1

        episode_scores = dict(
            zip(
                (config.get("final") or {}).get("names") or [],
                (config.get("final") or {}).get("scores") or [],
                strict=False,
            )
        )
        episodes.append(
            {
                "run_dir": str(run),
                "label": run.name,
                "seed": (config.get("seed_start") or [None])[0],
                "rotation": config.get("rotation"),
                "table": config.get("table") or list(specs),
                "specs": specs,
                "reviewed": bool(summary["games"]),
                "placements": (config.get("final") or {}).get("placements") or {},
                "scores": episode_scores,
            }
        )
        for spec, name in (
            (specs.get(name), name) for name in episode_scores
        ):
            if spec is None:
                continue
            acc = accum.setdefault(spec, _new_batch_accumulator(spec))
            others = [
                float(score)
                for other, score in episode_scores.items()
                if other != name
            ]
            if others:
                acc.setdefault("score_differentials", []).append(
                    float(episode_scores[name]) - sum(others) / len(others)
                )
                acc.setdefault("seed_values", {}).setdefault(
                    str((config.get("seed_start") or [None])[0]), []
                ).append(acc["score_differentials"][-1])

    engines = [_finalize_batch_engine(acc) for acc in accum.values()]
    engines.sort(
        key=lambda item: (
            float(item["avg_placement"]) if item["games"] else 99.0,
            -float(item["mean_rating"]),
            item["spec"],
        )
    )
    for index, engine in enumerate(engines):
        engine["rank"] = index + 1

    board = {
        "batch_dir": str(root),
        "generated": datetime.now(timezone.utc).isoformat(),
        "episodes": episodes,
        "leaderboard": engines,
        "engines": {engine["spec"]: engine for engine in engines},
        "episode_count": len(episodes),
        "reviewed_count": sum(1 for episode in episodes if episode["reviewed"]),
    }
    (root / "leaderboard.json").write_text(
        json.dumps(board, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return board


def _final_players(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Placement rows for an episode that finished but has not been graded yet, so a
    batch reads as standings before the Mortal review pass runs."""
    final = config.get("final") or {}
    names = [str(name) for name in final.get("names") or []]
    scores = list(final.get("scores") or [])
    placements = final.get("placements") or {}
    if not names or not placements:
        return []
    players = [
        {
            "seat": seat,
            "name": name,
            "score": _number_or_none(scores[seat] if seat < len(scores) else None),
            "placement": _int_or_none(placements.get(name)),
            "rating": None,
            "total_reviewed": 0,
            "total_matches": 0,
            "loss_sum": 0.0,
        }
        for seat, name in enumerate(names)
    ]
    return [{"players": players}]


def _new_batch_accumulator(spec: str) -> dict[str, Any]:
    acc = _new_accumulator(spec, spec)
    acc.update(
        {
            "names": set(),
            "table_positions": set(),
            "episodes": set(),
            "placement_counts": [0, 0, 0, 0],
            "decision_records": 0,
            "fallback_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": None,
            "latency_sum": 0.0,
            "latency_records": 0,
        }
    )
    return acc


def _finalize_batch_engine(acc: dict[str, Any]) -> dict[str, Any]:
    engine = _finalize_engine(acc)
    records = int(acc["decision_records"])
    engine.update(
        {
            "spec": acc["spec"],
            "names": sorted(acc["names"]),
            "table_positions": sorted(acc["table_positions"]),
            "episodes": len(acc["episodes"]),
            "placement_counts": list(acc["placement_counts"]),
            "decision_records": records,
            "fallback_count": int(acc["fallback_count"]),
            "fallback_rate": int(acc["fallback_count"]) / records if records else None,
            "input_tokens": int(acc["input_tokens"]),
            "output_tokens": int(acc["output_tokens"]),
            "cost": acc["cost"],
            "mean_latency_ms": (
                acc["latency_sum"] / acc["latency_records"]
                if acc["latency_records"]
                else None
            ),
            **_duplicate_metrics(acc),
        }
    )
    engine.pop("name", None)
    return engine


def _duplicate_metrics(acc: dict[str, Any]) -> dict[str, Any]:
    from math import sqrt
    from statistics import stdev

    diffs = [float(value) for value in acc.get("score_differentials") or []]
    block_means = [
        sum(values) / len(values)
        for values in (acc.get("seed_values") or {}).values()
        if values
    ]
    return {
        "avg_score_differential": _mean(diffs),
        "seed_block_mean": _mean(block_means),
        "standard_error": (
            stdev(block_means) / sqrt(len(block_means))
            if len(block_means) > 1
            else None
        ),
    }


def _new_accumulator(name: str, spec: str) -> dict[str, Any]:
    return {
        "name": name,
        "spec": spec,
        "games": 0,
        "placements": [],
        "scores": [],
        "ratings": [],
        "reviewed": 0,
        "matches": 0,
        "loss_sum": 0.0,
    }


def _add_player(acc: dict[str, Any], player: dict[str, Any]) -> None:
    acc["games"] += 1
    if player.get("placement") is not None:
        acc["placements"].append(float(player["placement"]))
    if player.get("score") is not None:
        acc["scores"].append(float(player["score"]))
    acc["ratings"].append(float(player.get("rating") or 0.0))
    reviewed = int(player.get("total_reviewed") or 0)
    acc["reviewed"] += reviewed
    acc["matches"] += int(player.get("total_matches") or 0)
    acc["loss_sum"] += float(player.get("loss_sum") or 0.0)


def _finalize_engine(acc: dict[str, Any]) -> dict[str, Any]:
    games = int(acc["games"])
    reviewed = int(acc["reviewed"])
    matches = int(acc["matches"])
    return {
        "name": acc["name"],
        "spec": acc["spec"],
        "games": games,
        "avg_placement": _mean(acc["placements"]),
        "avg_score": _mean(acc["scores"]),
        "mean_rating": _mean(acc["ratings"]),
        "match_rate": matches / reviewed if reviewed else 0.0,
        "mean_prob_loss": float(acc["loss_sum"]) / reviewed if reviewed else 0.0,
        "mean_q_weight_loss": float(acc["loss_sum"]) / reviewed if reviewed else 0.0,
        "total_reviewed": reviewed,
        "total_matches": matches,
    }


def _game_row(path: Path, raw_game: dict[str, Any]) -> dict[str, Any]:
    seed = raw_game.get("seed")
    if not isinstance(seed, list):
        seed = _seed_from_path(path)
    names = [str(name) for name in raw_game.get("names") or []]
    scores = list(raw_game.get("scores") or [])
    placements = raw_game.get("placements") or {}
    raw_players = raw_game.get("players") or {}
    players = []

    for seat in range(4):
        player = raw_players.get(str(seat)) or raw_players.get(seat) or {}
        name = str(
            player.get("name") or (names[seat] if seat < len(names) else f"P{seat}")
        )
        review = _review(player.get("review") or {})
        aggregates = _normal_aggregates(player.get("aggregates") or _aggregates(review))
        score = scores[seat] if seat < len(scores) else None
        placement = placements.get(name)
        total_reviewed = int(
            review.get("total_reviewed") or len(review.get("entries") or [])
        )
        total_matches = int(
            review.get("total_matches")
            if review.get("total_matches") is not None
            else sum(
                1 for entry in review.get("entries") or [] if entry.get("is_equal")
            )
        )
        loss_sum = sum(_prob_loss(entry) for entry in review.get("entries") or [])
        if not loss_sum and total_reviewed and aggregates.get("mean_prob_loss"):
            loss_sum = float(aggregates["mean_prob_loss"]) * total_reviewed
        players.append(
            {
                "seat": seat,
                "name": name,
                "score": _number_or_none(score),
                "placement": _int_or_none(placement),
                "rating": float(review.get("rating") or 0.0),
                "total_reviewed": total_reviewed,
                "total_matches": total_matches,
                "match_rate": total_matches / total_reviewed if total_reviewed else 0.0,
                "mean_prob_loss": loss_sum / total_reviewed if total_reviewed else 0.0,
                "mean_q_weight_loss": (
                    loss_sum / total_reviewed if total_reviewed else 0.0
                ),
                "loss_sum": loss_sum,
                "review": review,
                "aggregates": aggregates,
            }
        )

    return {
        "file": str(path),
        "seed": seed,
        "seed_label": _seed_label(seed),
        "names": names,
        "scores": [_number_or_none(score) for score in scores],
        "placements": placements,
        "placement_summary": _placement_summary(placements),
        "players": players,
    }


def _review(raw: dict[str, Any]) -> dict[str, Any]:
    review = dict(raw)
    review["entries"] = list(review.get("entries") or [])
    review["kyokus"] = list(review.get("kyokus") or [])
    review["rating"] = float(review.get("rating") or 0.0)
    review["total_reviewed"] = int(
        review.get("total_reviewed") or len(review["entries"])
    )
    review["total_matches"] = int(
        review.get("total_matches")
        if review.get("total_matches") is not None
        else sum(1 for entry in review["entries"] if entry.get("is_equal"))
    )
    return review


def _aggregates(review: dict[str, Any]) -> dict[str, Any]:
    entries = list(review.get("entries") or [])
    total_reviewed = int(review.get("total_reviewed") or len(entries))
    total_matches = int(
        review.get("total_matches")
        if review.get("total_matches") is not None
        else sum(1 for entry in entries if entry.get("is_equal"))
    )
    by_kind: dict[str, dict[str, Any]] = {
        kind: {"count": 0, "matches": 0, "mean_loss": 0.0}
        for kind in ["dahai", "reach", "chi", "pon", "kan", "hora", "ryukyoku", "none"]
    }
    loss_sums = {kind: 0.0 for kind in by_kind}
    losses = []
    for entry in entries:
        loss = _prob_loss(entry)
        losses.append(loss)
        kind = _kind(entry.get("actual") or {})
        if kind not in by_kind:
            by_kind[kind] = {"count": 0, "matches": 0, "mean_loss": 0.0}
            loss_sums[kind] = 0.0
        by_kind[kind]["count"] += 1
        by_kind[kind]["matches"] += int(bool(entry.get("is_equal")))
        loss_sums[kind] += loss
    for kind, stats in by_kind.items():
        count = int(stats["count"])
        stats["mean_loss"] = loss_sums[kind] / count if count else 0.0
    worst = [
        {
            "kyoku": entry.get("kyoku", 0),
            "honba": entry.get("honba", 0),
            "junme": entry.get("junme", 0),
            "actual": entry.get("actual") or {},
            "expected": entry.get("expected")
            or ((entry.get("details") or [{}])[0].get("event") or {}),
            "loss": loss,
        }
        for entry, loss in sorted(
            ((entry, _prob_loss(entry)) for entry in entries),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
    ]
    return {
        "match_rate": total_matches / total_reviewed if total_reviewed else 0.0,
        "mean_prob_loss": sum(losses) / len(losses) if losses else 0.0,
        "mean_q_weight_loss": sum(losses) / len(losses) if losses else 0.0,
        "worst": worst,
        "by_kind": by_kind,
    }


def _normal_aggregates(raw: dict[str, Any]) -> dict[str, Any]:
    aggregates = dict(raw)
    by_kind = {}
    for kind, stats in (aggregates.get("by_kind") or {}).items():
        item = dict(stats)
        if "matches" not in item and "matched" in item:
            item["matches"] = item["matched"]
        by_kind[str(kind)] = {
            "count": int(item.get("count") or 0),
            "matches": int(item.get("matches") or 0),
            "mean_loss": float(item.get("mean_loss") or 0.0),
        }
    aggregates["by_kind"] = by_kind
    aggregates["worst"] = list(aggregates.get("worst") or [])
    aggregates["match_rate"] = float(aggregates.get("match_rate") or 0.0)
    aggregates["mean_prob_loss"] = float(aggregates.get("mean_prob_loss") or 0.0)
    aggregates["mean_q_weight_loss"] = float(
        aggregates.get("mean_q_weight_loss", aggregates["mean_prob_loss"]) or 0.0
    )
    return aggregates


def _decision_stats(decisions_dir: Path, name: str) -> dict[str, Any]:
    path = decisions_dir / decision_filename(name)
    if not path.exists():
        legacy = decisions_dir / f"{name}.jsonl"
        try:
            legacy.resolve().relative_to(decisions_dir.resolve())
        except ValueError:
            pass
        else:
            if legacy.exists():
                path = legacy
    if not path.exists():
        return {
            "decision_records": 0,
            "fallback_count": 0,
            "fallback_rate": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": None,
            "mean_latency_ms": None,
        }

    records = 0
    fallback_count = 0
    input_tokens = 0
    output_tokens = 0
    cost: float | None = None
    latencies = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        records += 1
        fallback_count += int(bool(record.get("fallback")))
        usage = record.get("usage") or {}
        input_tokens += int(
            usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        )
        output_tokens += int(
            usage.get("output_tokens") or usage.get("completion_tokens") or 0
        )
        if usage.get("cost") is not None:
            cost = (cost or 0.0) + float(usage["cost"])
        latency = record.get("latency_ms")
        if isinstance(latency, int | float):
            latencies.append(float(latency))

    return {
        "decision_records": records,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / records if records else 0.0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
        "mean_latency_ms": _mean(latencies) if latencies else None,
    }


def _prob_loss(entry: dict[str, Any]) -> float:
    details = entry.get("details") or []
    if not details:
        return 0.0
    actual_index = int(entry.get("actual_index") or 0)
    if actual_index < 0 or actual_index >= len(details):
        actual_index = 0
    best = float((details[0] or {}).get("prob") or 0.0)
    actual = float((details[actual_index] or {}).get("prob") or 0.0)
    return max(0.0, best - actual)


def _kind(event: dict[str, Any]) -> str:
    event_type = str(event.get("type") or "none")
    if event_type in {"daiminkan", "ankan", "kakan"}:
        return "kan"
    return event_type


def _load_pai_sprite() -> str:
    root = Path(__file__).resolve().parents[1]
    path = root / "assets" / "pai.svg"
    if not path.exists():
        path = Path.cwd() / "assets" / "pai.svg"
    if path.exists():
        return _XML_RE.sub("", path.read_text(encoding="utf-8")).strip()

    page = Path(__file__).with_name("webui_page.html")
    if page.exists():
        match = _WEBUI_SPRITE_RE.search(page.read_text(encoding="utf-8"))
        if match is not None:
            return match.group(1).strip()
    raise FileNotFoundError("could not find the mahjong tile sprite")


def _sprite_view_box(sprite: str) -> str:
    match = re.search(
        r'<symbol\s+id=["\']tile["\'][^>]*\bviewBox=["\']([^"\']+)["\']', sprite
    )
    if match:
        return match.group(1)
    match = _VIEWBOX_RE.search(sprite)
    return match.group(1) if match else _TILE_VIEWBOX


def _html_document(label: str, sprite: str, data_json: str) -> str:
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>jongbench report \u2014 " + html.escape(label) + "</title>\n"
        "<style>\n" + _CSS + "\n</style>\n"
        "</head>\n"
        "<body>\n"
        '<div id="pai-assets" style="display:none" aria-hidden="true">\n'
        + sprite
        + "\n</div>\n"
        '<button class="theme-toggle" id="theme-toggle" type="button" aria-label="Toggle color theme" aria-pressed="false">Theme</button>\n'
        '<main class="page">\n'
        '<header class="card report-header">\n'
        '<div><p class="eyebrow">jongbench report</p><h1 id="report-label"></h1><p class="meta" id="report-meta"></p></div>\n'
        '<div class="models" id="models-list"></div>\n'
        "</header>\n"
        '<section class="stat-grid" id="stats"></section>\n'
        '<section class="card table-card"><div class="section-head"><h2>Leaderboard</h2><p id="review-status"></p></div><div class="table-wrap"><table id="leaderboard"></table></div></section>\n'
        '<section class="games" id="games"></section>\n'
        "</main>\n"
        '<script id="report-data" type="application/json">' + data_json + "</script>\n"
        "<script>\n" + _JS + "\n</script>\n"
        "</body>\n"
        "</html>\n"
    )


def _review_sort_key(path: Path) -> tuple[int, int, str]:
    seed = _seed_from_path(path)
    if len(seed) >= 2:
        return int(seed[0]), int(seed[1]), path.name
    return 0, 0, path.name


def _seed_from_path(path: Path) -> list[int]:
    parts = path.stem.split("_")
    if len(parts) >= 2:
        try:
            return [int(parts[0]), int(parts[1])]
        except ValueError:
            return [0, 0]
    return [0, 0]


def _seed_label(seed: Any) -> str:
    if isinstance(seed, list | tuple) and len(seed) >= 2:
        return f"{seed[0]}_{seed[1]}"
    return str(seed)


def _placement_summary(placements: dict[str, Any]) -> str:
    if not placements:
        return "placements unavailable"
    ordered = sorted(placements.items(), key=lambda item: int(item[1]))
    return ", ".join(f"{name} {_ordinal(int(place))}" for name, place in ordered)


def _ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _number_or_none(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


_CSS = r"""
:root {
  color-scheme: light;
  --surface:#fcfcfb;
  --page:#f9f9f7;
  --ink:#0b0b0b;
  --ink-2:#52514e;
  --muted:#898781;
  --grid:#e1e0d9;
  --baseline:#c3c2b7;
  --border:rgba(11,11,11,0.10);
  --accent:#2a78d6;
  --accent-deep:#1c5cab;
  --seq-250:#86b6ef;
  --good:#0ca30c;
  --serious:#ec835a;
  --critical:#d03b3b;
  --good-text:#006300;
  --series-0:#2a78d6;
  --series-1:#1baf7a;
  --series-2:#eda100;
  --series-3:#4a3aa7;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --surface:#1a1a19;
    --page:#0d0d0d;
    --ink:#ffffff;
    --ink-2:#c3c2b7;
    --muted:#898781;
    --grid:#2c2c2a;
    --baseline:#383835;
    --border:rgba(255,255,255,0.10);
    --accent:#3987e5;
    --accent-deep:#184f95;
    --seq-250:#86b6ef;
    --good:#0ca30c;
    --serious:#ec835a;
    --critical:#d03b3b;
    --good-text:#0ca30c;
    --series-0:#3987e5;
    --series-1:#199e70;
    --series-2:#c98500;
    --series-3:#9085e9;
  }
}
html[data-theme="light"] {
  color-scheme: light;
  --surface:#fcfcfb;
  --page:#f9f9f7;
  --ink:#0b0b0b;
  --ink-2:#52514e;
  --muted:#898781;
  --grid:#e1e0d9;
  --baseline:#c3c2b7;
  --border:rgba(11,11,11,0.10);
  --accent:#2a78d6;
  --accent-deep:#1c5cab;
  --seq-250:#86b6ef;
  --good:#0ca30c;
  --serious:#ec835a;
  --critical:#d03b3b;
  --good-text:#006300;
  --series-0:#2a78d6;
  --series-1:#1baf7a;
  --series-2:#eda100;
  --series-3:#4a3aa7;
}
html[data-theme="dark"] {
  color-scheme: dark;
  --surface:#1a1a19;
  --page:#0d0d0d;
  --ink:#ffffff;
  --ink-2:#c3c2b7;
  --muted:#898781;
  --grid:#2c2c2a;
  --baseline:#383835;
  --border:rgba(255,255,255,0.10);
  --accent:#3987e5;
  --accent-deep:#184f95;
  --seq-250:#86b6ef;
  --good:#0ca30c;
  --serious:#ec835a;
  --critical:#d03b3b;
  --good-text:#0ca30c;
  --series-0:#3987e5;
  --series-1:#199e70;
  --series-2:#c98500;
  --series-3:#9085e9;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--page);
  color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.45;
}
button, summary { font: inherit; }
button { color: var(--ink); }
.page {
  max-width: 1100px;
  margin: 0 auto;
  padding: 32px 18px 56px;
}
.card, .stat-card, details.player {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: 0 1px 2px rgba(0,0,0,.06);
}
.theme-toggle {
  position: fixed;
  top: 12px;
  right: 12px;
  z-index: 3;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface);
  padding: 7px 11px;
  cursor: pointer;
}
.theme-toggle:hover, .filter-row button:hover, summary:hover {
  background: rgba(42,120,214,.08);
}
.report-header {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(260px, 400px);
  gap: 24px;
  padding: 24px;
}
.eyebrow {
  margin: 0 0 6px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
}
h1, h2, h3, h4, p { margin-top: 0; }
h1 {
  margin-bottom: 8px;
  font-size: clamp(32px, 5vw, 56px);
  line-height: 1.02;
  letter-spacing: 0;
}
h2 { margin: 0; font-size: 20px; }
h3 { margin: 0; font-size: 16px; }
h4 { margin-bottom: 8px; font-size: 14px; }
.meta, .muted, .section-head p {
  color: var(--ink-2);
}
.models {
  display: grid;
  gap: 8px;
  align-content: start;
}
.model-row {
  display: grid;
  grid-template-columns: minmax(90px, 1fr) minmax(0, 1.4fr);
  gap: 12px;
  align-items: center;
  color: var(--ink-2);
  font-size: 13px;
}
.engine-name {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  color: var(--ink);
  font-weight: 650;
}
.engine-name span:last-child {
  min-width: 0;
  overflow-wrap: anywhere;
}
.swatch {
  flex: 0 0 auto;
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: var(--series);
  border: 1px solid var(--border);
}
.series-0 { --series: var(--series-0); }
.series-1 { --series: var(--series-1); }
.series-2 { --series: var(--series-2); }
.series-3 { --series: var(--series-3); }
.stat-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin: 16px 0;
}
.stat-card {
  padding: 16px;
  min-width: 0;
}
.stat-hero {
  margin: 12px 0 4px;
  color: var(--ink);
  font-size: 42px;
  font-weight: 760;
  line-height: 1;
}
.stat-sub {
  color: var(--ink-2);
  font-size: 13px;
}
.section-head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: baseline;
  padding: 18px 18px 0;
}
.table-card { overflow: hidden; }
.table-wrap {
  width: 100%;
  overflow-x: auto;
  padding: 12px 18px 18px;
}
table {
  width: 100%;
  border-collapse: collapse;
}
th, td {
  padding: 9px 8px;
  text-align: left;
  vertical-align: middle;
}
th {
  color: var(--ink-2);
  font-size: 12px;
  font-weight: 700;
  white-space: nowrap;
}
tbody tr + tr td {
  border-top: 1px solid var(--grid);
}
.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.games {
  display: grid;
  gap: 14px;
  margin-top: 16px;
}
details > summary {
  cursor: pointer;
  list-style: none;
}
details > summary::-webkit-details-marker { display: none; }
details.game > summary {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 16px 18px;
}
.summary-title {
  min-width: 0;
  overflow-wrap: anywhere;
  font-weight: 720;
}
.summary-meta {
  color: var(--ink-2);
  font-size: 13px;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.game-body {
  border-top: 1px solid var(--grid);
  padding: 16px 18px 18px;
}
.seat-table { margin-bottom: 14px; }
details.player {
  margin-top: 10px;
  box-shadow: none;
}
details.player > summary {
  padding: 12px 14px;
  font-weight: 680;
}
.player-body {
  border-top: 1px solid var(--grid);
  padding: 12px 14px 14px;
}
.filter-row {
  display: flex;
  gap: 8px;
  margin-bottom: 10px;
}
.filter-row button {
  border: 1px solid var(--border);
  border-radius: 999px;
  background: transparent;
  padding: 6px 10px;
  cursor: pointer;
}
.filter-row button[aria-pressed="true"] {
  border-color: var(--accent);
  background: rgba(42,120,214,.12);
}
.review-panel[data-filter="mistakes"] tr.is-match {
  display: none;
}
.review-table td {
  font-size: 13px;
}
.context-cell {
  white-space: nowrap;
}
.action-cell, .best-cell {
  min-width: 170px;
}
.action-inline {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  flex-wrap: wrap;
}
.action-label, .tag {
  color: var(--ink-2);
}
.tag {
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 1px 6px;
  font-size: 11px;
}
.pai {
  height: 1.6em;
  width: auto;
  vertical-align: middle;
}
.loss-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  border: 1px solid currentColor;
  border-radius: 999px;
  padding: 1px 7px;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.loss-chip.serious { color: var(--serious); }
.loss-chip.critical { color: var(--critical); }
.match-mark {
  color: var(--good-text);
  font-weight: 700;
}
.prob-cell {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 160px;
  min-width: 160px;
}
.prob-track {
  position: relative;
  flex: 1 1 auto;
  height: 8px;
  background: var(--grid);
  border-radius: 4px;
  box-shadow: inset 0 -1px 0 var(--baseline);
}
.prob-bar {
  position: absolute;
  left: 0;
  top: 0;
  height: 8px;
  border-radius: 4px;
  background: var(--accent);
}
.prob-marker {
  position: absolute;
  top: -2px;
  bottom: -2px;
  width: 2px;
  background: var(--accent-deep);
  transform: translateX(-1px);
}
.prob-value {
  width: 44px;
  color: var(--ink-2);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  text-align: right;
}
.strength {
  margin-top: 14px;
}
.strength > summary {
  display: inline-flex;
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 6px 10px;
  color: var(--ink);
}
.strength-body {
  display: grid;
  grid-template-columns: minmax(240px, .8fr) minmax(360px, 1.2fr);
  gap: 18px;
  margin-top: 12px;
}
.empty {
  padding: 22px;
  color: var(--ink-2);
}
@media (max-width: 820px) {
  .report-header, .strength-body {
    grid-template-columns: 1fr;
  }
  .stat-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .section-head, details.game > summary {
    align-items: flex-start;
    flex-direction: column;
  }
}
@media (max-width: 520px) {
  .page { padding-inline: 10px; }
  .stat-grid { grid-template-columns: 1fr; }
  .theme-toggle { position: absolute; }
}
"""


_JS = r"""
(function () {
  const data = JSON.parse(document.getElementById("report-data").textContent);
  const root = document.documentElement;
  const themeButton = document.getElementById("theme-toggle");
  const viewBox = String(data.tile_view_box || "0 0 320 446");
  let storedTheme = "";
  try {
    storedTheme = localStorage.getItem("jongbench-theme") || "";
  } catch (error) {
    storedTheme = "";
  }
  if (storedTheme === "dark" || storedTheme === "light") {
    root.setAttribute("data-theme", storedTheme);
  }
  function saveTheme(value) {
    try {
      if (value) {
        localStorage.setItem("jongbench-theme", value);
      } else {
        localStorage.removeItem("jongbench-theme");
      }
    } catch (error) {
    }
  }
  function syncThemeButton() {
    const theme = root.getAttribute("data-theme") || "";
    themeButton.setAttribute("aria-pressed", theme === "dark" ? "true" : "false");
    themeButton.textContent = theme === "dark" ? "Dark" : theme === "light" ? "Light" : "Theme";
  }
  themeButton.addEventListener("click", function () {
    const current = root.getAttribute("data-theme") || "";
    const next = current === "" ? "dark" : current === "dark" ? "light" : "";
    if (next) {
      root.setAttribute("data-theme", next);
    } else {
      root.removeAttribute("data-theme");
    }
    saveTheme(next);
    syncThemeButton();
  });
  syncThemeButton();

  const engines = data.leaderboard || [];
  const engineByName = new Map(engines.map(function (engine) { return [engine.name, engine]; }));
  renderHeader();
  renderStats();
  renderLeaderboard();
  renderGames();
  document.addEventListener("click", function (event) {
    const button = event.target.closest("[data-filter]");
    if (!button) {
      return;
    }
    const panel = button.closest(".review-panel");
    if (!panel) {
      return;
    }
    const filter = button.getAttribute("data-filter");
    panel.setAttribute("data-filter", filter);
    panel.querySelectorAll("[data-filter]").forEach(function (item) {
      item.setAttribute("aria-pressed", item === button ? "true" : "false");
    });
  });

  function renderHeader() {
    const config = data.config || {};
    const label = String(config.label || "run");
    document.getElementById("report-label").textContent = label;
    const created = config.created ? "created " + String(config.created) : "created date unavailable";
    const reviewed = String(data.review_files || 0) + " review file" + ((data.review_files || 0) === 1 ? "" : "s");
    document.getElementById("report-meta").textContent = created + " · " + reviewed;
    const names = config.names || engines.map(function (engine) { return engine.name; });
    const specs = config.models || [];
    document.getElementById("models-list").innerHTML = names.map(function (name, index) {
      return '<div class="model-row">' + engineName(name) + '<span>' + escapeHtml(specs[index] || "") + '</span></div>';
    }).join("");
  }

  function renderStats() {
    document.getElementById("stats").innerHTML = engines.map(function (engine) {
      return '<article class="stat-card">' +
        engineName(engine.name) +
        '<div class="stat-hero">' + fixed(engine.mean_rating * 100, 1) + '</div>' +
        '<div class="stat-sub">avg placement ' + fixed(engine.avg_placement, 2) + ' · match ' + percent(engine.match_rate, 0) + '</div>' +
        '</article>';
    }).join("");
  }

  function renderLeaderboard() {
    document.getElementById("review-status").textContent = data.missing_review_files ? String(data.missing_review_files) + " configured game(s) without review JSON" : "";
    const head = '<thead><tr>' +
      '<th class="num">rank</th><th>engine</th><th class="num">games</th><th class="num">avg placement</th>' +
      '<th class="num">avg score</th><th class="num">rating</th><th class="num">match %</th>' +
      '<th class="num">mean Q-weight loss</th><th class="num">fallbacks %</th><th class="num">tokens (in/out)</th><th class="num">cost</th><th class="num">mean latency</th>' +
      '</tr></thead>';
    const body = '<tbody>' + engines.map(function (engine) {
      return '<tr>' +
        '<td class="num">' + engine.rank + '</td>' +
        '<td>' + engineName(engine.name) + '</td>' +
        '<td class="num">' + integer(engine.games) + '</td>' +
        '<td class="num">' + fixed(engine.avg_placement, 2) + '</td>' +
        '<td class="num">' + integer(engine.avg_score) + '</td>' +
        '<td class="num">' + fixed(engine.mean_rating * 100, 1) + '</td>' +
        '<td class="num">' + percent(engine.match_rate, 0) + '</td>' +
        '<td class="num">' + percent(engine.mean_q_weight_loss, 1) + '</td>' +
        '<td class="num">' + nullablePercent(engine.fallback_rate, 1) + '</td>' +
        '<td class="num">' + tokenPair(engine) + '</td>' +
        '<td class="num">' + nullableCost(engine.cost) + '</td>' +
        '<td class="num">' + nullableMs(engine.mean_latency_ms) + '</td>' +
        '</tr>';
    }).join("") + '</tbody>';
    document.getElementById("leaderboard").innerHTML = head + body;
  }

  function renderGames() {
    const games = data.games || [];
    const target = document.getElementById("games");
    if (!games.length) {
      target.innerHTML = '<section class="card empty">No review files found in review/.</section>';
      return;
    }
    target.innerHTML = games.map(function (game) {
      const reviewed = game.players.reduce(function (total, player) { return total + Number(player.total_reviewed || 0); }, 0);
      return '<details class="card game">' +
        '<summary><span class="summary-title">Game ' + escapeHtml(game.seed_label) + ' — ' + escapeHtml(game.placement_summary) + '</span><span class="summary-meta">' + integer(reviewed) + ' reviewed decisions</span></summary>' +
        '<div class="game-body">' + seatTable(game) + game.players.map(playerBlock).join("") + '</div>' +
        '</details>';
    }).join("");
  }

  function seatTable(game) {
    const rows = game.players.map(function (player) {
      return '<tr>' +
        '<td class="num">P' + player.seat + '</td>' +
        '<td>' + engineName(player.name) + '</td>' +
        '<td class="num">' + integer(player.score) + '</td>' +
        '<td class="num">' + ordinal(player.placement) + '</td>' +
        '<td class="num">' + fixed(player.rating * 100, 1) + '</td>' +
        '<td class="num">' + String(player.total_matches) + '/' + String(player.total_reviewed) + '</td>' +
        '</tr>';
    }).join("");
    return '<div class="table-wrap seat-table"><table><thead><tr><th class="num">seat</th><th>engine</th><th class="num">score</th><th class="num">place</th><th class="num">rating</th><th class="num">matched</th></tr></thead><tbody>' + rows + '</tbody></table></div>';
  }

  function playerBlock(player) {
    return '<details class="player">' +
      '<summary>P' + player.seat + ' ' + escapeHtml(player.name) + ' — value rating ' + fixed(player.rating * 100, 1) + ', value matched ' + String(player.total_matches) + '/' + String(player.total_reviewed) + policySummary(player) + '</summary>' +
      '<div class="player-body">' + decisionTable(player) + strengths(player) + '</div>' +
      '</details>';
  }

  function policySummary(player) {
    const aggregates = player.aggregates || {};
    if (!Number(aggregates.policy_count || 0) || aggregates.policy_match_rate == null) {
      return "";
    }
    return ', policy matched ' + percent(aggregates.policy_match_rate, 1);
  }

  function decisionTable(player) {
    const entries = (player.review && player.review.entries) || [];
    if (!entries.length) {
      return '<p class="muted">No reviewed decisions for this player.</p>';
    }
    const rows = entries.map(function (entry) {
      const isMatch = Boolean(entry.is_equal);
      const best = (entry.details || [])[0] || {};
      const policyBest = entry.policy_expected;
      const hasPolicy = Boolean(policyBest);
      return '<tr class="' + (isMatch ? "is-match" : "is-mistake") + '">' +
        '<td class="num">' + kyokuLabel(entry.kyoku, entry.honba) + ' / ' + integer(entry.junme) + '</td>' +
        '<td class="context-cell">' + tileHtml(entry.tile) + '</td>' +
        '<td class="action-cell">' + actionHtml(entry.actual) + ' ' + decisionMark(entry) + '</td>' +
        '<td class="best-cell">' + actionHtml(best.event || entry.expected) + '</td>' +
        '<td class="best-cell">' + (hasPolicy ? actionHtml(policyBest) : '<span class="muted">—</span>') + '</td>' +
        '<td>' + probCell(actualProb(entry), Number(best.prob || 0)) + '</td>' +
        '<td>' + (hasPolicy ? probCell(actualPolicyProb(entry), bestPolicyProb(entry)) : '<span class="muted">—</span>') + '</td>' +
        '</tr>';
    }).join("");
    return '<div class="review-panel" data-filter="mistakes">' +
      '<div class="filter-row"><button type="button" data-filter="mistakes" aria-pressed="true">Mistakes only</button><button type="button" data-filter="all" aria-pressed="false">All decisions</button></div>' +
      '<div class="table-wrap"><table class="review-table"><thead><tr><th class="num">kyoku/junme</th><th>context tile</th><th>actual action</th><th>value best</th><th>policy best</th><th>value weight</th><th>policy probability</th></tr></thead><tbody>' + rows + '</tbody></table></div>' +
      '</div>';
  }

  function strengths(player) {
    const aggregates = player.aggregates || {};
    const byKind = aggregates.by_kind || {};
    const kindRows = Object.keys(byKind).sort().map(function (kind) {
      const stats = byKind[kind] || {};
      return '<tr><td>' + escapeHtml(kind) + '</td><td class="num">' + integer(stats.count) + '</td><td class="num">' + integer(stats.matches) + '</td><td class="num">' + percent(stats.mean_loss, 1) + '</td></tr>';
    }).join("");
    const worst = aggregates.worst || [];
    const worstRows = worst.map(function (entry) {
      return '<tr>' +
        '<td class="num">' + kyokuLabel(entry.kyoku, entry.honba) + ' / ' + integer(entry.junme) + '</td>' +
        '<td>' + actionHtml(entry.actual) + '</td>' +
        '<td>' + actionHtml(entry.expected) + '</td>' +
        '<td>' + lossChip(Number(entry.loss || 0)) + '</td>' +
        '</tr>';
    }).join("");
    return '<details class="strength">' +
      '<summary>Strengths &amp; weaknesses</summary>' +
      '<div class="strength-body">' +
      '<section><h4>By kind</h4><div class="table-wrap"><table><thead><tr><th>kind</th><th class="num">count</th><th class="num">matched</th><th class="num">mean loss</th></tr></thead><tbody>' + kindRows + '</tbody></table></div></section>' +
      '<section><h4>Worst 10</h4><div class="table-wrap"><table><thead><tr><th class="num">kyoku/junme</th><th>actual</th><th>expected</th><th>loss</th></tr></thead><tbody>' + worstRows + '</tbody></table></div></section>' +
      '</div></details>';
  }

  function engineName(name) {
    const engine = engineByName.get(name) || {};
    const index = Number.isInteger(engine.series_index) ? engine.series_index : 0;
    return '<span class="engine-name series-' + Math.max(0, Math.min(3, index)) + '"><span class="swatch" aria-hidden="true"></span><span>' + escapeHtml(name) + '</span></span>';
  }

  function tileHtml(tile) {
    if (!tile) {
      return '<span class="muted">—</span>';
    }
    const raw = String(tile);
    const id = raw.toLowerCase();
    if (!/^[0-9a-z]+$/.test(id)) {
      return '<span>' + escapeHtml(raw) + '</span>';
    }
    return '<svg class="pai" viewBox="' + escapeAttr(viewBox) + '" role="img" aria-label="' + escapeAttr(raw) + '"><use href="#pai-' + escapeAttr(id) + '"></use></svg>';
  }

  function actionHtml(event) {
    if (!event || !event.type) {
      return '<span class="action-inline"><span class="action-label">none</span></span>';
    }
    const type = String(event.type);
    if (type === "none") {
      return '<span class="action-inline"><span class="action-label">pass</span></span>';
    }
    if (type === "dahai") {
      return '<span class="action-inline">' + tileHtml(event.pai) + '<span class="action-label">discard</span>' + (event.tsumogiri ? '<span class="tag">tsumogiri</span>' : "") + '</span>';
    }
    if (type === "reach") {
      return '<span class="action-inline"><span class="action-label">reach</span></span>';
    }
    if (type === "hora") {
      return '<span class="action-inline"><span class="action-label">agari</span></span>';
    }
    if (type === "ryukyoku") {
      return '<span class="action-inline"><span class="action-label">ryukyoku</span></span>';
    }
    if (type === "chi" || type === "pon" || type === "daiminkan" || type === "ankan" || type === "kakan") {
      const consumed = Array.isArray(event.consumed) ? event.consumed.map(tileHtml).join("") : "";
      return '<span class="action-inline"><span class="action-label">' + escapeHtml(type) + '</span>' + (event.pai ? tileHtml(event.pai) : "") + consumed + '</span>';
    }
    return '<span class="action-inline"><span class="action-label">' + escapeHtml(type) + '</span></span>';
  }

  function decisionMark(entry) {
    if (entry.is_equal) {
      return '<span class="match-mark" aria-label="matched">✓</span>';
    }
    return lossChip(probLoss(entry));
  }

  function lossChip(loss) {
    const level = loss < 0.35 ? "serious" : "critical";
    return '<span class="loss-chip ' + level + '"><span aria-hidden="true">\u25be</span><span>-' + fixed(loss * 100, 1) + '%</span><span>' + level + '</span></span>';
  }

  function probCell(actual, best) {
    const actualPct = clamp(actual, 0, 1) * 100;
    const bestPct = clamp(best, 0, 1) * 100;
    return '<div class="prob-cell"><div class="prob-track"><div class="prob-bar" style="width:' + actualPct.toFixed(2) + '%"></div><span class="prob-marker" style="left:' + bestPct.toFixed(2) + '%"></span></div><span class="prob-value">' + percent(actual, 0) + '</span></div>';
  }

  function actualProb(entry) {
    const details = entry.details || [];
    if (!details.length) {
      return 0;
    }
    let index = Number(entry.actual_index || 0);
    if (index < 0 || index >= details.length) {
      index = 0;
    }
    return Number(details[index].prob || 0);
  }

  function probLoss(entry) {
    const details = entry.details || [];
    if (!details.length) {
      return 0;
    }
    return Math.max(0, Number(details[0].prob || 0) - actualProb(entry));
  }

  function actualPolicyProb(entry) {
    if (entry.policy_actual_prob != null) {
      return Number(entry.policy_actual_prob);
    }
    const details = entry.details || [];
    let index = Number(entry.actual_index || 0);
    if (index < 0 || index >= details.length) {
      return 0;
    }
    return Number(details[index].policy_prob || 0);
  }

  function bestPolicyProb(entry) {
    return (entry.details || []).reduce(function (best, detail) {
      return Math.max(best, Number(detail.policy_prob || 0));
    }, 0);
  }

  function kyokuLabel(kyoku, honba) {
    const winds = ["E", "S", "W", "N"];
    const value = Number(kyoku || 0);
    const wind = winds[Math.floor(value / 4)] || "E";
    const hand = value % 4 + 1;
    const suffix = Number(honba || 0) ? "-" + String(honba) : "";
    return wind + String(hand) + suffix;
  }

  function fixed(value, digits) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return "—";
    }
    return number.toFixed(digits);
  }

  function integer(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return "—";
    }
    return Math.round(number).toLocaleString();
  }

  function percent(value, digits) {
    return fixed(Number(value || 0) * 100, digits) + "%";
  }

  function nullablePercent(value, digits) {
    return value === null || value === undefined ? "—" : percent(value, digits);
  }

  function nullableMs(value) {
    return value === null || value === undefined ? "—" : fixed(value, 0) + " ms";
  }

  function nullableCost(value) {
    return value === null || value === undefined ? "—" : "$" + fixed(value, 4);
  }

  function tokenPair(engine) {
    if (!engine.decision_records) {
      return "—";
    }
    return integer(engine.input_tokens) + " / " + integer(engine.output_tokens);
  }

  function ordinal(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return "—";
    }
    const mod100 = number % 100;
    const suffix = mod100 >= 10 && mod100 <= 20 ? "th" : ({1: "st", 2: "nd", 3: "rd"}[number % 10] || "th");
    return String(number) + suffix;
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, Number(value) || 0));
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, function (char) {
      return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[char];
    });
  }

  function escapeAttr(value) {
    return escapeHtml(value);
  }
})();
"""
