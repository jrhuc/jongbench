"""Strict conversion from a finished hanchan artifact to a replay capsule."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .capsule import ReplayCapsule, ScriptedDecision, SEATS
from .schema import (
    _boolean,
    _integer,
    _number,
    _sha256,
    _string,
    _string_tuple,
)

_MISSING = object()


def _read_journal(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number}: journal row must be an object")
        rows.append(value)
    if not rows:
        raise ValueError("episode journal is empty")
    if rows[0].get("journal") != 2:
        raise ValueError("unsupported episode journal version")
    end_indexes = [index for index, row in enumerate(rows) if row.get("end") is True]
    if not end_indexes:
        raise ValueError("episode journal is incomplete (missing end marker)")
    if end_indexes != [len(rows) - 1]:
        raise ValueError("episode journal end marker must be the final row")
    return rows


def _coherent(
    config: dict[str, Any],
    header: dict[str, Any],
    key: str,
    *,
    default: object = _MISSING,
) -> object:
    config_value = config.get(key, _MISSING)
    header_value = header.get(key, _MISSING)
    if (
        config_value is not _MISSING
        and header_value is not _MISSING
        and config_value != header_value
    ):
        raise ValueError(f"journal {key} does not match config {key}")
    if config_value is not _MISSING:
        return config_value
    if header_value is not _MISSING:
        return header_value
    if default is not _MISSING:
        return default
    raise ValueError(f"episode is missing {key}")


def _checkpoint_sha(config: dict[str, Any], header: dict[str, Any]) -> str | None:
    values: list[str | None] = []
    for where, container in (("config", config), ("journal", header)):
        checkpoint = container.get("control_checkpoint", _MISSING)
        if checkpoint is _MISSING:
            continue
        if checkpoint is None:
            values.append(None)
            continue
        if not isinstance(checkpoint, dict):
            raise TypeError(f"{where}.control_checkpoint must be an object or null")
        values.append(
            _sha256(
                checkpoint.get("sha256"),
                f"{where}.control_checkpoint.sha256",
            )
        )
    if len(values) == 2 and values[0] != values[1]:
        raise ValueError("journal control checkpoint does not match config")
    return values[0] if values else None


def _outcomes(
    config: dict[str, Any],
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    final = config.get("final")
    if not isinstance(final, dict):
        raise TypeError("config.final must be a JSON object")
    names = _string_tuple(final.get("names"), "config.final.names", length=4)
    raw_scores = final.get("scores")
    if not isinstance(raw_scores, list) or len(raw_scores) != 4:
        raise TypeError("config.final.scores must contain four integers")
    scores = tuple(
        (name, _integer(score, f"config.final.scores[{index}]"))
        for index, (name, score) in enumerate(zip(names, raw_scores, strict=True))
    )
    raw_placements = final.get("placements")
    if not isinstance(raw_placements, dict):
        raise TypeError("config.final.placements must be a JSON object")
    placements = tuple(
        (
            name,
            _integer(
                raw_placements.get(name), f"config.final.placements[{name!r}]"
            ),
        )
        for name in SEATS
    )
    return scores, placements


def load_episode_capsule(episode_dir: str | Path) -> ReplayCapsule:
    root = Path(episode_dir)
    config_path = root / "config.json"
    journal_path = root / "journal.jsonl"
    if not config_path.is_file():
        raise FileNotFoundError(f"episode config not found: {config_path}")
    if not journal_path.is_file():
        raise FileNotFoundError(f"episode journal not found: {journal_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("episode config must be a JSON object")
    rows = _read_journal(journal_path)
    header = rows[0]

    seed_raw = config.get("seed_start")
    if not isinstance(seed_raw, list) or len(seed_raw) != 2:
        seed_raw = [header.get("seed"), 1]
    seed = (
        _integer(seed_raw[0], "config.seed_start[0]"),
        _integer(seed_raw[1], "config.seed_start[1]"),
    )
    if _integer(header.get("seed"), "journal.seed") != seed[0]:
        raise ValueError("journal seed does not match config.seed_start")

    rotation = _integer(_coherent(config, header, "rotation"), "rotation")
    table = _string_tuple(config.get("table"), "config.table", length=4)
    names = _string_tuple(config.get("names"), "config.names", length=4)
    model_specs = _string_tuple(config.get("models"), "config.models", length=4)
    header_models = header.get("models", _MISSING)
    if header_models is not _MISSING:
        recorded = _string_tuple(header_models, "journal.models", length=4)
        if recorded != model_specs:
            raise ValueError("journal models do not match config models")
    models = tuple(zip(names, model_specs, strict=True))

    profile = _string(_coherent(config, header, "profile"), "profile")
    evaluated_raw = _coherent(config, header, "evaluated_agent", default=None)
    evaluated = (
        None
        if evaluated_raw is None
        else _string(evaluated_raw, "evaluated_agent")
    )
    control_use_policy = _boolean(
        _coherent(config, header, "control_use_policy", default=False),
        "control_use_policy",
    )
    control_epsilon = _number(
        _coherent(config, header, "control_boltzmann_epsilon", default=0.0),
        "control_boltzmann_epsilon",
    )
    control_temp = _number(
        _coherent(config, header, "control_boltzmann_temp", default=1.0),
        "control_boltzmann_temp",
    )

    final_scores, final_placements = _outcomes(config)
    decisions: list[ScriptedDecision] = []
    for row in rows[1:-1]:
        if row.get("checkpoint") is True:
            continue
        if "seat" not in row:
            raise ValueError("unexpected journal row without a seat")
        decisions.append(ScriptedDecision.from_journal_row(len(decisions), row))
    if not decisions:
        raise ValueError("episode journal contains no model-mediated decisions")

    return ReplayCapsule(
        seed=seed,
        rotation=rotation,
        table=(table[0], table[1], table[2], table[3]),
        models=models,
        evaluated_seat=evaluated,
        profile=profile,
        control_checkpoint_sha256=_checkpoint_sha(config, header),
        control_use_policy=control_use_policy,
        control_boltzmann_epsilon=control_epsilon,
        control_boltzmann_temp=control_temp,
        final_scores=final_scores,
        final_placements=final_placements,
        decisions=tuple(decisions),
        source=str(root),
    )
