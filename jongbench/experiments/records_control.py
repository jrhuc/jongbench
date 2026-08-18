"""Typed records for all-control baselines and policy replacements."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Self

from .identity import ControlPolicyIdentity
from .schema import (
    _integer,
    _number,
    _optional_string,
    _string,
    content_id,
)


def _scores4(values: object, field: str) -> tuple[float, float, float, float]:
    if not isinstance(values, list) or len(values) != 4:
        raise TypeError(f"{field} must contain four numbers")
    numbers = tuple(
        _number(value, f"{field}[{index}]")
        for index, value in enumerate(values)
    )
    return (numbers[0], numbers[1], numbers[2], numbers[3])


def _placements4(values: object, field: str) -> tuple[int, int, int, int]:
    if not isinstance(values, list) or len(values) != 4:
        raise TypeError(f"{field} must contain four integers")
    placements = tuple(
        _integer(value, f"{field}[{index}]")
        for index, value in enumerate(values)
    )
    if sorted(placements) != [1, 2, 3, 4]:
        raise ValueError(f"{field} must be a permutation of 1, 2, 3, 4")
    return (placements[0], placements[1], placements[2], placements[3])


@dataclass(frozen=True, slots=True)
class AllControlBaseline:
    """One deterministic all-control outcome, indexed by physical chair."""

    record_id: str
    baseline_key: str
    wall_id: str
    control: ControlPolicyIdentity
    scores: tuple[float, float, float, float]
    placements: tuple[int, int, int, int]
    log_path: str | None = None

    def __post_init__(self) -> None:
        if self.baseline_key != self._expected_key():
            raise ValueError("baseline_key does not match wall and control policy")
        if any(not math.isfinite(value) for value in self.scores):
            raise ValueError("baseline scores must be finite")
        if sorted(self.placements) != [1, 2, 3, 4]:
            raise ValueError(
                "baseline placements must be a permutation of 1, 2, 3, 4"
            )
        if self.record_id != self._expected_record_id():
            raise ValueError("baseline record_id does not match canonical content")

    @staticmethod
    def key_for(wall_id: str, control: ControlPolicyIdentity) -> str:
        return content_id(
            "all-control-baseline-key-v1",
            {"wall_id": wall_id, "control_policy_id": control.policy_id},
        )

    @classmethod
    def create(
        cls,
        *,
        wall_id: str,
        control: ControlPolicyIdentity,
        scores: tuple[float, float, float, float],
        placements: tuple[int, int, int, int],
        log_path: str | None = None,
    ) -> Self:
        baseline_key = cls.key_for(wall_id, control)
        record_id = content_id(
            "all-control-baseline-record-v1",
            {
                "baseline_key": baseline_key,
                "scores": list(scores),
                "placements": list(placements),
            },
        )
        return cls(
            record_id=record_id,
            baseline_key=baseline_key,
            wall_id=wall_id,
            control=control,
            scores=scores,
            placements=placements,
            log_path=log_path,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        if value.get("record_type") != "all_control_baseline":
            raise ValueError("not an all-control baseline record")
        return cls(
            record_id=_string(value.get("record_id"), "record_id"),
            baseline_key=_string(value.get("baseline_key"), "baseline_key"),
            wall_id=_string(value.get("wall_id"), "wall_id"),
            control=ControlPolicyIdentity.from_dict(value.get("control")),
            scores=_scores4(value.get("scores"), "scores"),
            placements=_placements4(value.get("placements"), "placements"),
            log_path=_optional_string(value.get("log_path"), "log_path"),
        )

    def score(self, table_position: int) -> float:
        if table_position not in range(4):
            raise ValueError("table_position must be in [0, 3]")
        return self.scores[table_position]

    def placement(self, table_position: int) -> int:
        if table_position not in range(4):
            raise ValueError("table_position must be in [0, 3]")
        return self.placements[table_position]

    def as_dict(self) -> dict[str, object]:
        return {
            "record_type": "all_control_baseline",
            "record_id": self.record_id,
            "baseline_key": self.baseline_key,
            "wall_id": self.wall_id,
            "control": self.control.as_dict(),
            "scores": list(self.scores),
            "placements": list(self.placements),
            "log_path": self.log_path,
        }

    def _expected_key(self) -> str:
        return self.key_for(self.wall_id, self.control)

    def _expected_record_id(self) -> str:
        return content_id(
            "all-control-baseline-record-v1",
            {
                "baseline_key": self.baseline_key,
                "scores": list(self.scores),
                "placements": list(self.placements),
            },
        )


@dataclass(frozen=True, slots=True)
class MatchedControlResult:
    """Policy replacement minus all-control outcome on one wall and chair."""

    record_id: str
    pair_id: str
    capsule_id: str
    baseline_record_id: str
    wall_id: str
    profile: str
    control: ControlPolicyIdentity
    model: str
    seat: str
    table_position: int
    observed_score: float
    control_score: float
    score_delta: float
    observed_placement: int
    control_placement: int
    log_path: str | None = None

    def __post_init__(self) -> None:
        if self.pair_id != self._expected_pair_id():
            raise ValueError("pair_id does not match canonical experiment key")
        if self.table_position not in range(4):
            raise ValueError("table_position must be in [0, 3]")
        if self.observed_placement not in range(1, 5):
            raise ValueError("observed_placement must be in [1, 4]")
        if self.control_placement not in range(1, 5):
            raise ValueError("control_placement must be in [1, 4]")
        values = (self.observed_score, self.control_score, self.score_delta)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("matched-control scores must be finite")
        if not math.isclose(
            self.score_delta,
            self.observed_score - self.control_score,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("score_delta does not match observed minus control")
        if self.record_id != self._expected_record_id():
            raise ValueError(
                "matched-control record_id does not match canonical content"
            )

    @classmethod
    def create(
        cls,
        *,
        capsule_id: str,
        baseline_record_id: str,
        wall_id: str,
        profile: str,
        control: ControlPolicyIdentity,
        model: str,
        seat: str,
        table_position: int,
        observed_score: float,
        control_score: float,
        observed_placement: int,
        control_placement: int,
        log_path: str | None = None,
    ) -> Self:
        pair_id = content_id(
            "matched-control-pair-v1",
            {
                "capsule_id": capsule_id,
                "baseline_record_id": baseline_record_id,
                "seat": seat,
                "table_position": table_position,
            },
        )
        score_delta = observed_score - control_score
        record_id = content_id(
            "matched-control-record-v1",
            {
                "pair_id": pair_id,
                "wall_id": wall_id,
                "profile": profile,
                "control_policy_id": control.policy_id,
                "model": model,
                "observed_score": observed_score,
                "control_score": control_score,
                "score_delta": score_delta,
                "observed_placement": observed_placement,
                "control_placement": control_placement,
            },
        )
        return cls(
            record_id=record_id,
            pair_id=pair_id,
            capsule_id=capsule_id,
            baseline_record_id=baseline_record_id,
            wall_id=wall_id,
            profile=profile,
            control=control,
            model=model,
            seat=seat,
            table_position=table_position,
            observed_score=observed_score,
            control_score=control_score,
            score_delta=score_delta,
            observed_placement=observed_placement,
            control_placement=control_placement,
            log_path=log_path,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        if value.get("record_type") != "matched_control_result":
            raise ValueError("not a matched-control result record")
        return cls(
            record_id=_string(value.get("record_id"), "record_id"),
            pair_id=_string(value.get("pair_id"), "pair_id"),
            capsule_id=_string(value.get("capsule_id"), "capsule_id"),
            baseline_record_id=_string(
                value.get("baseline_record_id"), "baseline_record_id"
            ),
            wall_id=_string(value.get("wall_id"), "wall_id"),
            profile=_string(value.get("profile"), "profile"),
            control=ControlPolicyIdentity.from_dict(value.get("control")),
            model=_string(value.get("model"), "model"),
            seat=_string(value.get("seat"), "seat"),
            table_position=_integer(value.get("table_position"), "table_position"),
            observed_score=_number(value.get("observed_score"), "observed_score"),
            control_score=_number(value.get("control_score"), "control_score"),
            score_delta=_number(value.get("score_delta"), "score_delta"),
            observed_placement=_integer(
                value.get("observed_placement"), "observed_placement"
            ),
            control_placement=_integer(
                value.get("control_placement"), "control_placement"
            ),
            log_path=_optional_string(value.get("log_path"), "log_path"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "record_type": "matched_control_result",
            "record_id": self.record_id,
            "pair_id": self.pair_id,
            "capsule_id": self.capsule_id,
            "baseline_record_id": self.baseline_record_id,
            "wall_id": self.wall_id,
            "profile": self.profile,
            "control": self.control.as_dict(),
            "model": self.model,
            "seat": self.seat,
            "table_position": self.table_position,
            "observed_score": self.observed_score,
            "control_score": self.control_score,
            "score_delta": self.score_delta,
            "observed_placement": self.observed_placement,
            "control_placement": self.control_placement,
            "log_path": self.log_path,
        }

    def _expected_pair_id(self) -> str:
        return content_id(
            "matched-control-pair-v1",
            {
                "capsule_id": self.capsule_id,
                "baseline_record_id": self.baseline_record_id,
                "seat": self.seat,
                "table_position": self.table_position,
            },
        )

    def _expected_record_id(self) -> str:
        return content_id(
            "matched-control-record-v1",
            {
                "pair_id": self.pair_id,
                "wall_id": self.wall_id,
                "profile": self.profile,
                "control_policy_id": self.control.policy_id,
                "model": self.model,
                "observed_score": self.observed_score,
                "control_score": self.control_score,
                "score_delta": self.score_delta,
                "observed_placement": self.observed_placement,
                "control_placement": self.control_placement,
            },
        )
