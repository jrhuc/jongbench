from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .run_artifacts import log_sort_key


def print_summary(summary: dict[str, Any]) -> None:
    engines = list(summary.get("leaderboard") or [])
    headers = [
        "engine",
        "games",
        "place",
        "score",
        "rating",
        "match",
        "fallbacks",
        "cost",
    ]
    rows = []
    for engine in engines:
        fallback_rate = engine.get("fallback_rate")
        if fallback_rate is None:
            fallbacks = "n/a"
        else:
            fallbacks = f"{float(fallback_rate) * 100:.1f}% ({int(engine.get('fallback_count') or 0)})"
        rows.append(
            [
                str(engine.get("name") or ""),
                str(int(engine.get("games") or 0)),
                f"{float(engine.get('avg_placement') or 0.0):.2f}",
                f"{float(engine.get('avg_score') or 0.0):.0f}",
                f"{float(engine.get('mean_rating') or 0.0) * 100:.2f}",
                f"{float(engine.get('match_rate') or 0.0) * 100:.1f}%",
                fallbacks,
                _cost(engine.get("cost")),
            ]
        )

    print_table(headers, rows)


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        if rows
        else len(headers[index])
        for index in range(len(headers))
    ]
    print(
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    )
    for row in rows:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def print_leaderboard(board: dict[str, Any]) -> None:
    print(
        f"{board['episode_count']} episode(s), {board['reviewed_count']} reviewed"
        f" -> {board['batch_dir']}"
    )
    headers = [
        "#",
        "spec",
        "eps",
        "place",
        "1/2/3/4",
        "score",
        "rating",
        "match",
        "fallbacks",
        "cost",
    ]
    rows = []
    for engine in board.get("leaderboard") or []:
        fallback_rate = engine.get("fallback_rate")
        fallbacks = (
            "n/a"
            if fallback_rate is None
            else f"{float(fallback_rate) * 100:.1f}% ({int(engine.get('fallback_count') or 0)})"
        )
        reviewed = int(engine.get("total_reviewed") or 0)
        rows.append(
            [
                str(engine.get("rank") or ""),
                str(engine.get("spec") or ""),
                str(int(engine.get("episodes") or 0)),
                f"{float(engine.get('avg_placement') or 0.0):.2f}",
                "/".join(str(count) for count in engine.get("placement_counts") or []),
                f"{float(engine.get('avg_score') or 0.0):.0f}",
                f"{float(engine.get('mean_rating') or 0.0) * 100:.2f}"
                if reviewed
                else "n/a",
                f"{float(engine.get('match_rate') or 0.0) * 100:.1f}%"
                if reviewed
                else "n/a",
                fallbacks,
                _cost(engine.get("cost")),
            ]
        )
    print_table(headers, rows)


def _cost(value: Any) -> str:
    # Only metering providers report a cost; a local or unmetered seat has none.
    return "n/a" if value is None else f"${float(value):.4f}"


def print_seat_ratings(run_dir: Path) -> None:
    paths = sorted((run_dir / "review").glob("*.json"), key=log_sort_key)
    if not paths:
        return
    data = json.loads(paths[0].read_text(encoding="utf-8"))
    players = data.get("players") or {}
    print("seat ratings")
    for seat in range(4):
        player = players.get(str(seat)) or players.get(seat) or {}
        review_data = player.get("review") or {}
        rating = float(review_data.get("rating") or 0.0) * 100.0
        name = player.get("name") or f"P{seat}"
        print(f"P{seat} {name!s:<24} {rating:6.2f}")
