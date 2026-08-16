"""Content-addressed identities shared by experiment artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Self

from .schema import _boolean, _number, _sha256, _string, content_id


@dataclass(frozen=True, slots=True)
class ControlPolicyIdentity:
    """Everything that makes two fixed control policies experimentally distinct."""

    policy_id: str
    checkpoint_sha256: str
    use_policy: bool
    boltzmann_epsilon: float
    boltzmann_temp: float

    def __post_init__(self) -> None:
        _sha256(self.checkpoint_sha256, "control.checkpoint_sha256")
        epsilon = float(self.boltzmann_epsilon)
        temperature = float(self.boltzmann_temp)
        if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("control.boltzmann_epsilon must be finite and in [0, 1]")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("control.boltzmann_temp must be finite and positive")
        if self.policy_id != self._expected_id():
            raise ValueError("control.policy_id does not match canonical content")

    @classmethod
    def create(
        cls,
        *,
        checkpoint_sha256: str,
        use_policy: bool,
        boltzmann_epsilon: float,
        boltzmann_temp: float,
    ) -> Self:
        payload = {
            "checkpoint_sha256": checkpoint_sha256,
            "use_policy": bool(use_policy),
            "boltzmann_epsilon": float(boltzmann_epsilon),
            "boltzmann_temp": float(boltzmann_temp),
        }
        return cls(
            policy_id=content_id("control-policy-v1", payload),
            checkpoint_sha256=checkpoint_sha256,
            use_policy=bool(use_policy),
            boltzmann_epsilon=float(boltzmann_epsilon),
            boltzmann_temp=float(boltzmann_temp),
        )

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if not isinstance(value, dict):
            raise TypeError("control must be a JSON object")
        return cls(
            policy_id=_string(value.get("policy_id"), "control.policy_id"),
            checkpoint_sha256=_sha256(
                value.get("checkpoint_sha256"), "control.checkpoint_sha256"
            ),
            use_policy=_boolean(value.get("use_policy"), "control.use_policy"),
            boltzmann_epsilon=_number(
                value.get("boltzmann_epsilon"), "control.boltzmann_epsilon"
            ),
            boltzmann_temp=_number(
                value.get("boltzmann_temp"), "control.boltzmann_temp"
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "use_policy": self.use_policy,
            "boltzmann_epsilon": self.boltzmann_epsilon,
            "boltzmann_temp": self.boltzmann_temp,
        }

    def _expected_id(self) -> str:
        return content_id(
            "control-policy-v1",
            {
                "checkpoint_sha256": self.checkpoint_sha256,
                "use_policy": self.use_policy,
                "boltzmann_epsilon": self.boltzmann_epsilon,
                "boltzmann_temp": self.boltzmann_temp,
            },
        )
