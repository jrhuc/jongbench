"""Typed records for forced actions on one realized hidden wall."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Self

from .identity import ControlPolicyIdentity
from .schema import _integer, _number, _optional_string, _string, content_id


@dataclass(frozen=True, slots=True)
class BranchResult:
    """One forced action on the original hidden wall.

    This is a causal result for one realized world. It is not expected value
    conditional on the player's information set.
    """

    result_id: str
    capsule_id: str
    opportunity_id: str
    decision_id: str
    wall_id: str
    profile: str
    control: ControlPolicyIdentity
    estimand: str
    target_sequence: int
    seat: str
    table_position: int
    kyoku: int
    honba: int
    junme: int
    action_index: int
    action_count: int
    action_label: str
    model_choice: int
    reference_choice: int | None
    point_delta: float
    final_score: float
    placement: int
    log_path: str | None = None

    def __post_init__(self) -> None:
        if self.estimand != "factual_wall":
            raise ValueError("unsupported branch estimand")
        if self.target_sequence < 0:
            raise ValueError("target_sequence must be non-negative")
        if self.action_count < 2:
            raise ValueError("action_count must be at least two")
        if self.action_index not in range(self.action_count):
            raise ValueError("action_index is out of range")
        if self.model_choice not in range(self.action_count):
            raise ValueError("model_choice is out of range")
        if self.reference_choice is not None and self.reference_choice not in range(
            self.action_count
        ):
            raise ValueError("reference_choice is out of range")
        if self.table_position not in range(4):
            raise ValueError("table_position must be in [0, 3]")
        if self.placement not in range(1, 5):
            raise ValueError("placement must be in [1, 4]")
        if not math.isfinite(self.point_delta) or not math.isfinite(self.final_score):
            raise ValueError("branch scores must be finite")
        if self.result_id != self._expected_result_id():
            raise ValueError("branch result_id does not match canonical content")

    @classmethod
    def create(
        cls,
        *,
        capsule_id: str,
        opportunity_id: str,
        decision_id: str,
        wall_id: str,
        profile: str,
        control: ControlPolicyIdentity,
        estimand: str,
        target_sequence: int,
        seat: str,
        table_position: int,
        kyoku: int,
        honba: int,
        junme: int,
        action_index: int,
        action_count: int,
        action_label: str,
        model_choice: int,
        reference_choice: int | None,
        point_delta: float,
        final_score: float,
        placement: int,
        log_path: str | None = None,
    ) -> Self:
        payload = {
            "capsule_id": capsule_id,
            "opportunity_id": opportunity_id,
            "decision_id": decision_id,
            "wall_id": wall_id,
            "profile": profile,
            "control_policy_id": control.policy_id,
            "estimand": estimand,
            "target_sequence": target_sequence,
            "seat": seat,
            "table_position": table_position,
            "kyoku": kyoku,
            "honba": honba,
            "junme": junme,
            "action_index": action_index,
            "action_count": action_count,
            "action_label": action_label,
            "model_choice": model_choice,
            "reference_choice": reference_choice,
            "point_delta": point_delta,
            "final_score": final_score,
            "placement": placement,
        }
        return cls(
            result_id=content_id("branch-result-v1", payload),
            capsule_id=capsule_id,
            opportunity_id=opportunity_id,
            decision_id=decision_id,
            wall_id=wall_id,
            profile=profile,
            control=control,
            estimand=estimand,
            target_sequence=target_sequence,
            seat=seat,
            table_position=table_position,
            kyoku=kyoku,
            honba=honba,
            junme=junme,
            action_index=action_index,
            action_count=action_count,
            action_label=action_label,
            model_choice=model_choice,
            reference_choice=reference_choice,
            point_delta=point_delta,
            final_score=final_score,
            placement=placement,
            log_path=log_path,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        if value.get("record_type") != "branch_result":
            raise ValueError("not a branch result record")
        reference = value.get("reference_choice")
        return cls(
            result_id=_string(value.get("result_id"), "result_id"),
            capsule_id=_string(value.get("capsule_id"), "capsule_id"),
            opportunity_id=_string(value.get("opportunity_id"), "opportunity_id"),
            decision_id=_string(value.get("decision_id"), "decision_id"),
            wall_id=_string(value.get("wall_id"), "wall_id"),
            profile=_string(value.get("profile"), "profile"),
            control=ControlPolicyIdentity.from_dict(value.get("control")),
            estimand=_string(value.get("estimand"), "estimand"),
            target_sequence=_integer(value.get("target_sequence"), "target_sequence"),
            seat=_string(value.get("seat"), "seat"),
            table_position=_integer(value.get("table_position"), "table_position"),
            kyoku=_integer(value.get("kyoku"), "kyoku"),
            honba=_integer(value.get("honba"), "honba"),
            junme=_integer(value.get("junme"), "junme"),
            action_index=_integer(value.get("action_index"), "action_index"),
            action_count=_integer(value.get("action_count"), "action_count"),
            action_label=_string(value.get("action_label"), "action_label"),
            model_choice=_integer(value.get("model_choice"), "model_choice"),
            reference_choice=(
                None if reference is None else _integer(reference, "reference_choice")
            ),
            point_delta=_number(value.get("point_delta"), "point_delta"),
            final_score=_number(value.get("final_score"), "final_score"),
            placement=_integer(value.get("placement"), "placement"),
            log_path=_optional_string(value.get("log_path"), "log_path"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "record_type": "branch_result",
            "result_id": self.result_id,
            **self._semantic_payload(),
            "control": self.control.as_dict(),
            "log_path": self.log_path,
        }

    def _semantic_payload(self) -> dict[str, object]:
        return {
            "capsule_id": self.capsule_id,
            "opportunity_id": self.opportunity_id,
            "decision_id": self.decision_id,
            "wall_id": self.wall_id,
            "profile": self.profile,
            "estimand": self.estimand,
            "target_sequence": self.target_sequence,
            "seat": self.seat,
            "table_position": self.table_position,
            "kyoku": self.kyoku,
            "honba": self.honba,
            "junme": self.junme,
            "action_index": self.action_index,
            "action_count": self.action_count,
            "action_label": self.action_label,
            "model_choice": self.model_choice,
            "reference_choice": self.reference_choice,
            "point_delta": self.point_delta,
            "final_score": self.final_score,
            "placement": self.placement,
        }

    def _expected_result_id(self) -> str:
        return content_id(
            "branch-result-v1",
            {
                **self._semantic_payload(),
                "control_policy_id": self.control.policy_id,
            },
        )
