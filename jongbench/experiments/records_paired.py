"""Typed records for paired prompt, rendering, hint, and tool arms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Self

from .schema import _boolean, _number, _string, content_id


@dataclass(frozen=True, slots=True)
class PairedArmObservation:
    """One scalar from one member of a controlled paired intervention."""

    observation_id: str
    pair_id: str
    cluster_id: str
    profile: str
    experiment: str
    arm: str
    metric: str
    value: float
    valid: bool = True

    def __post_init__(self) -> None:
        for field, value in (
            ("pair_id", self.pair_id),
            ("cluster_id", self.cluster_id),
            ("profile", self.profile),
            ("experiment", self.experiment),
            ("arm", self.arm),
            ("metric", self.metric),
        ):
            if not value:
                raise ValueError(f"{field} must be non-empty")
        if not math.isfinite(self.value):
            raise ValueError("paired-arm value must be finite")
        if self.observation_id != self._expected_id():
            raise ValueError("observation_id does not match canonical content")

    @classmethod
    def create(
        cls,
        *,
        pair_id: str,
        cluster_id: str,
        profile: str,
        experiment: str,
        arm: str,
        metric: str,
        value: float,
        valid: bool = True,
    ) -> Self:
        payload = {
            "pair_id": pair_id,
            "cluster_id": cluster_id,
            "profile": profile,
            "experiment": experiment,
            "arm": arm,
            "metric": metric,
            "value": value,
            "valid": bool(valid),
        }
        return cls(
            observation_id=content_id("paired-arm-observation-v1", payload),
            pair_id=pair_id,
            cluster_id=cluster_id,
            profile=profile,
            experiment=experiment,
            arm=arm,
            metric=metric,
            value=value,
            valid=bool(valid),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        if value.get("record_type") != "paired_arm_observation":
            raise ValueError("not a paired-arm observation record")
        return cls(
            observation_id=_string(value.get("observation_id"), "observation_id"),
            pair_id=_string(value.get("pair_id"), "pair_id"),
            cluster_id=_string(value.get("cluster_id"), "cluster_id"),
            profile=_string(value.get("profile"), "profile"),
            experiment=_string(value.get("experiment"), "experiment"),
            arm=_string(value.get("arm"), "arm"),
            metric=_string(value.get("metric"), "metric"),
            value=_number(value.get("value"), "value"),
            valid=_boolean(value.get("valid", True), "valid"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "record_type": "paired_arm_observation",
            "observation_id": self.observation_id,
            "pair_id": self.pair_id,
            "cluster_id": self.cluster_id,
            "profile": self.profile,
            "experiment": self.experiment,
            "arm": self.arm,
            "metric": self.metric,
            "value": self.value,
            "valid": self.valid,
        }

    def _expected_id(self) -> str:
        return content_id(
            "paired-arm-observation-v1",
            {
                "pair_id": self.pair_id,
                "cluster_id": self.cluster_id,
                "profile": self.profile,
                "experiment": self.experiment,
                "arm": self.arm,
                "metric": self.metric,
                "value": self.value,
                "valid": self.valid,
            },
        )
