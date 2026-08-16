"""JSONL I/O for typed experiment facts."""

from __future__ import annotations

import json
from pathlib import Path

from .records_branch import BranchResult
from .records_control import AllControlBaseline, MatchedControlResult
from .records_paired import PairedArmObservation
from .schema import _canonical_json

__all__ = [
    "AllControlBaseline",
    "BranchResult",
    "MatchedControlResult",
    "PairedArmObservation",
    "ExperimentRow",
    "read_jsonl",
    "write_jsonl",
]


type ExperimentRow = (
    AllControlBaseline | BranchResult | MatchedControlResult | PairedArmObservation
)


def write_jsonl(path: str | Path, rows: list[object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            as_dict = getattr(row, "as_dict", None)
            if not callable(as_dict):
                raise TypeError(f"row {type(row).__name__} has no as_dict()")
            handle.write(_canonical_json(as_dict()) + "\n")
    return destination


def read_jsonl(path: str | Path) -> list[ExperimentRow]:
    source = Path(path)
    rows: list[ExperimentRow] = []
    constructors = {
        "all_control_baseline": AllControlBaseline.from_dict,
        "branch_result": BranchResult.from_dict,
        "matched_control_result": MatchedControlResult.from_dict,
        "paired_arm_observation": PairedArmObservation.from_dict,
    }
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{source}:{line_number}: record must be a JSON object")
        record_type = value.get("record_type")
        constructor = constructors.get(record_type)
        if constructor is None:
            raise ValueError(
                f"{source}:{line_number}: unsupported record_type {record_type!r}"
            )
        rows.append(constructor(value))
    return rows
