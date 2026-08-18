from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jongbench.artifacts import decision_filename
from jongbench.report import _WEBUI_SPRITE_RE, leaderboard, summarize, write_report


NAMES = ["modelA", "modelB", "modelC", "modelD"]


def test_packaged_webui_contains_report_sprite_fallback() -> None:
    page_path = ROOT / "jongbench" / "webui_page.html"
    if not page_path.exists():
        pytest.skip("webui_page.html not built; run `cd webui && bun run build`")
    page = page_path.read_text(encoding="utf-8")
    match = _WEBUI_SPRITE_RE.search(page)
    assert match is not None
    assert 'id="pai-1m"' in match.group(1)


def test_report_generation() -> None:
    with tempfile.TemporaryDirectory(prefix="jongbench-report-") as tempdir:
        run = Path(tempdir)
        (run / "review").mkdir()
        (run / "decisions").mkdir()
        (run / "config.json").write_text(
            json.dumps(
                {
                    "label": "synthetic",
                    "created": "2026-07-09T12:00:00Z",
                    "models": ["specA", "specB", "specC", "specD"],
                    "names": NAMES,
                    "games": 2,
                    "seed_start": [7000, 3],
                }
            ),
            encoding="utf-8",
        )
        _write_game(run, [7000, 3], NAMES, [42000, 30000, 18000, 10000])
        _write_game(
            run,
            [7001, 3],
            ["modelB", "modelC", "modelD", "modelA"],
            [35000, 31000, 22000, 12000],
        )
        _write_decisions(run)

        summary = summarize(str(run))
        report_path = Path(write_report(str(run)))
        text = report_path.read_text(encoding="utf-8")
        summary_json = json.loads((run / "summary.json").read_text(encoding="utf-8"))

        assert report_path.exists()
        assert '<use href="#pai-' in text
        for name in NAMES:
            assert name in text
        assert "data-theme" in text
        assert "Mistakes only" in text
        assert len(summary_json["engines"]) == 4
        assert all(
            isinstance(engine["mean_rating"], float)
            for engine in summary_json["engines"].values()
        )
        assert "{{" not in text
        assert report_path.stat().st_size < 2_500_000
        assert set(summary["engines"]) == set(NAMES)

    print("OK")


def test_leaderboard_pools_a_batch_by_spec() -> None:
    specs = ["specA", "specB", "specC", "specD"]
    with tempfile.TemporaryDirectory(prefix="jongbench-batch-") as tempdir:
        batch = Path(tempdir)
        # The same four specs under different engine names in each episode: pooling has to
        # follow the spec, not the name.
        _write_episode(
            batch,
            "hanchan-00000",
            [8000, 1],
            NAMES,
            specs,
            [42000, 30000, 18000, 10000],
            graded=True,
        )
        _write_episode(
            batch,
            "hanchan-00001",
            [8001, 1],
            ["a", "b", "c", "d"],
            specs,
            [12000, 22000, 31000, 35000],
            graded=True,
        )
        _write_episode(
            batch,
            "hanchan-00002",
            [8002, 1],
            ["w", "x", "y", "z"],
            specs,
            [40000, 30000, 20000, 10000],
            graded=False,
        )

        board = leaderboard(str(batch))
        engines = board["engines"]

        assert board["episode_count"] == 3
        assert board["reviewed_count"] == 2
        assert (
            json.loads((batch / "leaderboard.json").read_text(encoding="utf-8"))
            == board
        )
        assert set(engines) == set(specs)
        assert all(engine["episodes"] == 3 for engine in engines.values())
        assert engines["specA"]["names"] == ["a", "modelA", "w"]
        assert engines["specA"]["placement_counts"] == [2, 0, 0, 1]
        assert engines["specD"]["placement_counts"] == [1, 0, 0, 2]
        # Only the two graded episodes carry ratings; the ungraded one must not dilute them.
        assert all(engine["total_reviewed"] == 6 for engine in engines.values())
        assert engines["specA"]["mean_rating"] > 0.0
        assert [engine["spec"] for engine in board["leaderboard"]] == sorted(
            specs, key=lambda spec: engines[spec]["avg_placement"]
        )
        assert [engine["rank"] for engine in board["leaderboard"]] == [1, 2, 3, 4]


def test_leaderboard_pools_metered_cost_across_episodes() -> None:
    specs = ["specA", "specB", "specC", "specD"]
    with tempfile.TemporaryDirectory(prefix="jongbench-cost-") as tempdir:
        batch = Path(tempdir)
        _write_episode(
            batch,
            "hanchan-00000",
            [8000, 1],
            NAMES,
            specs,
            [40000, 30000, 20000, 10000],
            graded=False,
        )
        _write_episode(
            batch,
            "hanchan-00001",
            [8001, 1],
            ["a", "b", "c", "d"],
            specs,
            [40000, 30000, 20000, 10000],
            graded=False,
        )
        # specA is metered under a different engine name in each episode; specB is a
        # local seat that never reports a cost.
        _write_decision_log(batch / "hanchan-00000", "modelA", [0.01, 0.002])
        _write_decision_log(batch / "hanchan-00001", "a", [0.003])
        _write_decision_log(batch / "hanchan-00000", "modelB", [None, None])

        engines = leaderboard(str(batch))["engines"]

        assert engines["specA"]["cost"] == pytest.approx(0.015)
        assert engines["specB"]["cost"] is None
        assert engines["specC"]["cost"] is None


def _write_decision_log(run: Path, name: str, costs: list[float | None]) -> None:
    records = []
    for cost in costs:
        usage: dict[str, Any] = {"input_tokens": 100, "output_tokens": 20}
        if cost is not None:
            usage["cost"] = cost
        records.append({"usage": usage, "fallback": False, "latency_ms": 900})
    (run / "decisions" / decision_filename(name)).write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_episode(
    batch: Path,
    label: str,
    seed: list[int],
    names: list[str],
    specs: list[str],
    scores: list[int],
    *,
    graded: bool,
) -> None:
    run = batch / label
    (run / "review").mkdir(parents=True)
    (run / "decisions").mkdir()
    placements = {
        name: index + 1
        for index, name in enumerate(
            name for _, name in sorted(zip(scores, names), reverse=True)
        )
    }
    (run / "config.json").write_text(
        json.dumps(
            {
                "label": label,
                "created": "2026-08-11T12:00:00Z",
                "models": specs,
                "names": names,
                "games": 1,
                "seed_start": seed,
                "rotation": 0,
                "final": {"names": names, "scores": scores, "placements": placements},
            }
        ),
        encoding="utf-8",
    )
    if graded:
        _write_game(run, seed, names, scores)


def _write_game(
    run: Path, seed: list[int], names: list[str], scores: list[int]
) -> None:
    placements = {
        name: index + 1
        for index, name in enumerate(
            name for _, name in sorted(zip(scores, names), reverse=True)
        )
    }
    players = {}
    for seat, name in enumerate(names):
        rating = 0.58 + seat * 0.04 + (0.04 if name == "modelA" else 0.0)
        review = _review(seat, rating)
        players[str(seat)] = {
            "name": name,
            "review": review,
            "aggregates": _aggregates(review),
        }
    (run / "review" / f"{seed[0]}_{seed[1]}.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "names": names,
                "scores": scores,
                "placements": placements,
                "players": players,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def _review(seat: int, rating: float) -> dict[str, Any]:
    entries = [
        _entry(
            kyoku=0,
            junme=3,
            tile="5m",
            actual={"type": "dahai", "actor": seat, "pai": "5m", "tsumogiri": True},
            expected={"type": "dahai", "actor": seat, "pai": "5m", "tsumogiri": True},
            alternatives=[
                (
                    {"type": "dahai", "actor": seat, "pai": "5m", "tsumogiri": True},
                    1.4,
                    0.62,
                ),
                (
                    {"type": "dahai", "actor": seat, "pai": "8p", "tsumogiri": False},
                    0.8,
                    0.38,
                ),
            ],
            actual_index=0,
            is_equal=True,
        ),
        _entry(
            kyoku=1,
            junme=8,
            tile="3p",
            actual={"type": "dahai", "actor": seat, "pai": "2m", "tsumogiri": False},
            expected={"type": "dahai", "actor": seat, "pai": "3p", "tsumogiri": True},
            alternatives=[
                (
                    {"type": "dahai", "actor": seat, "pai": "3p", "tsumogiri": True},
                    2.1,
                    0.72,
                ),
                (
                    {"type": "dahai", "actor": seat, "pai": "2m", "tsumogiri": False},
                    0.4,
                    0.18,
                ),
                ({"type": "none"}, 0.1, 0.10),
            ],
            actual_index=1,
            is_equal=False,
        ),
        _entry(
            kyoku=5,
            junme=11,
            tile="5mr",
            actual={
                "type": "ankan",
                "actor": seat,
                "consumed": ["5m", "5m", "5m", "5mr"],
            },
            expected={
                "type": "ankan",
                "actor": seat,
                "consumed": ["5m", "5m", "5m", "5mr"],
            },
            alternatives=[
                (
                    {
                        "type": "ankan",
                        "actor": seat,
                        "consumed": ["5m", "5m", "5m", "5mr"],
                    },
                    1.2,
                    0.55,
                ),
                (
                    {"type": "dahai", "actor": seat, "pai": "9s", "tsumogiri": False},
                    0.7,
                    0.45,
                ),
            ],
            actual_index=0,
            is_equal=True,
        ),
    ]
    return {
        "rating": rating,
        "total_reviewed": len(entries),
        "total_matches": sum(1 for entry in entries if entry["is_equal"]),
        "temperature": 0.1,
        "entries": entries,
        "kyokus": [],
    }


def _entry(
    *,
    kyoku: int,
    junme: int,
    tile: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
    alternatives: list[tuple[dict[str, Any], float, float]],
    actual_index: int,
    is_equal: bool,
) -> dict[str, Any]:
    assert abs(sum(prob for _, _, prob in alternatives) - 1.0) < 1e-9
    return {
        "kyoku": kyoku,
        "honba": 0,
        "junme": junme,
        "tiles_left": 70 - junme,
        "last_actor": actual.get("actor", 0),
        "tile": tile,
        "actual": actual,
        "expected": expected,
        "is_equal": is_equal,
        "actual_index": actual_index,
        "shanten": 1,
        "at_furiten": False,
        "details": [
            {"event": event, "q_value": q_value, "prob": prob}
            for event, q_value, prob in alternatives
        ],
    }


def _aggregates(review: dict[str, Any]) -> dict[str, Any]:
    entries = list(review["entries"])
    losses = [_loss(entry) for entry in entries]
    by_kind: dict[str, dict[str, Any]] = {}
    for entry, loss in zip(entries, losses, strict=True):
        kind = _kind(entry["actual"])
        stats = by_kind.setdefault(kind, {"count": 0, "matches": 0, "mean_loss": 0.0})
        stats["count"] += 1
        stats["matches"] += int(entry["is_equal"])
        stats["mean_loss"] += loss
    for stats in by_kind.values():
        stats["mean_loss"] /= stats["count"]
    worst = [
        {
            "kyoku": entry["kyoku"],
            "honba": entry["honba"],
            "junme": entry["junme"],
            "actual": entry["actual"],
            "expected": entry["expected"],
            "loss": loss,
        }
        for entry, loss in sorted(
            zip(entries, losses, strict=True), key=lambda item: item[1], reverse=True
        )[:10]
    ]
    return {
        "match_rate": review["total_matches"] / review["total_reviewed"],
        "mean_prob_loss": sum(losses) / len(losses),
        "worst": worst,
        "by_kind": by_kind,
    }


def _loss(entry: dict[str, Any]) -> float:
    details = entry["details"]
    return max(0.0, details[0]["prob"] - details[entry["actual_index"]]["prob"])


def _kind(event: dict[str, Any]) -> str:
    if event["type"] in {"ankan", "kakan", "daiminkan"}:
        return "kan"
    return str(event["type"])


def _write_decisions(run: Path) -> None:
    records = [
        {
            "usage": {"input_tokens": 120, "output_tokens": 30},
            "fallback": False,
            "latency_ms": 950,
        },
        {
            "usage": {"input_tokens": 100, "output_tokens": 25},
            "fallback": True,
            "latency_ms": 1250,
        },
    ]
    (run / "decisions" / "modelA.jsonl").write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


if __name__ == "__main__":
    test_report_generation()
