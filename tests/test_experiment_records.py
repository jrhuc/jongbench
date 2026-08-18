from __future__ import annotations

import json
from pathlib import Path

import pytest

from jongbench.experiments.capsule import ReplayCapsule


def _episode(root: Path) -> Path:
    episode = root / "hanchan-00000"
    episode.mkdir(parents=True)
    table = ["seat2", "seat3", "seat0", "seat1"]
    profile = "sha256:" + "1" * 64
    checkpoint = {
        "path": "/cache/mortal.pth",
        "sha256": "a" * 64,
        "source": "fixture",
        "use_policy": False,
    }
    config = {
        "names": ["seat0", "seat1", "seat2", "seat3"],
        "models": ["model/a", "model/b", "mortal", "model/c"],
        "seed_start": [20260000, 1],
        "rotation": 2,
        "table": table,
        "evaluated_agent": "seat0",
        "profile": profile,
        "control_use_policy": False,
        "control_boltzmann_epsilon": 0.0,
        "control_boltzmann_temp": 1.0,
        "control_checkpoint": checkpoint,
        "final": {
            "names": table,
            "scores": [26000, 24000, 30000, 20000],
            "placements": {"seat0": 1, "seat1": 4, "seat2": 2, "seat3": 3},
        },
    }
    (episode / "config.json").write_text(json.dumps(config), encoding="utf-8")
    rows = [
        {
            "journal": 2,
            "seed": 20260000,
            "rotation": 2,
            "models": ["model/a", "model/b", "mortal", "model/c"],
            "evaluated_agent": "seat0",
            "profile": profile,
            "control_use_policy": False,
            "control_boltzmann_epsilon": 0.0,
            "control_boltzmann_temp": 1.0,
            "control_checkpoint": checkpoint,
        },
        {
            "seat": "seat0",
            "player_id": 2,
            "kyoku": 0,
            "honba": 0,
            "junme": 1,
            "tiles_left": 69,
            "kyoku_events_len": 2,
            "menu": ["discard 1m", "discard 2m"],
            "choice": 1,
            "choice_label": "discard 2m",
            "fallback": None,
        },
        {"checkpoint": True},
        {
            "seat": "seat3",
            "player_id": 1,
            "kyoku": 0,
            "honba": 0,
            "junme": 2,
            "tiles_left": 66,
            "kyoku_events_len": 7,
            "menu": ["pass", "pon 3m"],
            "choice": 0,
            "choice_label": "pass",
            "fallback": None,
        },
        {
            "seat": "seat1",
            "player_id": 3,
            "kyoku": 0,
            "honba": 0,
            "junme": 2,
            "tiles_left": 66,
            "kyoku_events_len": 7,
            "menu": ["pass", "ron"],
            "choice": 0,
            "choice_label": "pass",
            "fallback": None,
        },
        {
            "seat": "seat0",
            "player_id": 2,
            "kyoku": 0,
            "honba": 0,
            "junme": 3,
            "tiles_left": 61,
            "kyoku_events_len": 12,
            "menu": ["discard 3p", "discard 4p", "discard 5p"],
            "choice": 2,
            "choice_label": "discard 5p",
            "fallback": None,
        },
        {"end": True},
    ]
    (episode / "journal.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return episode


def test_capsule_is_content_addressed_and_portable(tmp_path: Path) -> None:
    first = ReplayCapsule.from_episode(_episode(tmp_path / "one"))
    second = ReplayCapsule.from_episode(_episode(tmp_path / "two"))

    assert first.capsule_id == second.capsule_id
    assert first.source != second.source
    assert first.table == ("seat2", "seat3", "seat0", "seat1")
    assert first.table_position("seat0") == 2
    assert first.score("seat0") == 30000
    assert first.placement("seat0") == 1
    assert first.model("seat0") == "model/a"
    assert [decision.sequence for decision in first.decisions] == [0, 1, 2, 3]
    assert first.wall_id == second.wall_id


def test_capsule_keeps_cotemporal_reactions_in_the_intervention_prefix(
    tmp_path: Path,
) -> None:
    capsule = ReplayCapsule.from_episode(_episode(tmp_path))
    scripts = capsule.scripts_by_seat(1, forced_choice=1)

    assert [row.sequence for row in scripts["seat0"]] == [0]
    assert [row.sequence for row in scripts["seat3"]] == [1]
    assert [row.sequence for row in scripts["seat1"]] == [2]
    assert scripts["seat3"][0].choice == 1
    assert all(row.sequence <= 2 for rows in scripts.values() for row in rows)
    assert capsule.opportunity_id(1) == capsule.opportunity_id(2)
    assert capsule.opportunity_id(0) != capsule.opportunity_id(1)


def test_capsule_round_trip_preserves_identity(tmp_path: Path) -> None:
    capsule = ReplayCapsule.from_episode(_episode(tmp_path))
    loaded = ReplayCapsule.read(capsule.write(tmp_path / "capsule.json"))
    assert loaded == capsule
    assert loaded.capsule_id == capsule.capsule_id


def test_capsule_rejects_a_journal_menu_choice_mismatch(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    rows = [
        json.loads(line)
        for line in (episode / "journal.jsonl").read_text().splitlines()
    ]
    rows[1]["choice"] = 7
    (episode / "journal.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="out of range"):
        ReplayCapsule.from_episode(episode)


def test_capsule_rejects_an_unfinished_episode(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    lines = (episode / "journal.jsonl").read_text().splitlines()
    (episode / "journal.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="missing end marker"):
        ReplayCapsule.from_episode(episode)


def test_capsule_rejects_rows_after_the_end_marker(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    with (episode / "journal.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"checkpoint": True}) + "\n")
    with pytest.raises(ValueError, match="end marker must be the final row"):
        ReplayCapsule.from_episode(episode)


def test_capsule_rejects_config_journal_identity_drift(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    lines = (episode / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["models"] = ["other", "model/b", "mortal", "model/c"]
    lines[0] = json.dumps(header)
    (episode / "journal.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="models do not match"):
        ReplayCapsule.from_episode(episode)


def test_capsule_rejects_control_checkpoint_drift(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    lines = (episode / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["control_checkpoint"]["sha256"] = "b" * 64
    lines[0] = json.dumps(header)
    (episode / "journal.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="control checkpoint"):
        ReplayCapsule.from_episode(episode)
