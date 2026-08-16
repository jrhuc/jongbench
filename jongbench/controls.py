"""Canonical construction of fixed Mortal control seats.

The runtime, matched-control experiments, and self-play all need the same opponent
identity.  Loading a checkpoint once and sharing its frozen network modules across
fresh arena-seat wrappers keeps that identity explicit and avoids loading the same
weights once per seat or branch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    import torch

from .weights import AUTO_MORTAL_WEIGHTS, CheckpointInput, ResolvedCheckpoint


@dataclass(frozen=True, slots=True)
class MortalControlConfig:
    """Everything that changes a Mortal control policy.

    ``boltzmann_epsilon`` is part of the policy identity.  Counterfactual replay
    currently requires it to be zero because the arena does not yet couple the
    control's sampling stream across branches.
    """

    weights: CheckpointInput = AUTO_MORTAL_WEIGHTS
    use_policy: bool = False
    boltzmann_epsilon: float = 0.0
    boltzmann_temp: float = 1.0
    device: str | torch.device | None = None
    enable_quick_eval: bool = True
    enable_rule_based_agari_guard: bool = True

    def __post_init__(self) -> None:
        epsilon = float(self.boltzmann_epsilon)
        temperature = float(self.boltzmann_temp)
        if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("boltzmann_epsilon must be finite and in [0, 1]")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("boltzmann_temp must be finite and positive")


def clone_mortal_engine(template: Any, name: str, **overrides: Any) -> Any:
    """Clone one inference facade while sharing the template's frozen modules."""
    from .mortal_engine import MortalEngine

    values: dict[str, Any] = {
        "is_oracle": template.is_oracle,
        "version": template.version,
        "device": template.device,
        "stochastic_latent": template.stochastic_latent,
        "enable_amp": template.enable_amp,
        "enable_quick_eval": template.enable_quick_eval,
        "enable_rule_based_agari_guard": template.enable_rule_based_agari_guard,
        "name": name,
        "boltzmann_epsilon": template.boltzmann_epsilon,
        "boltzmann_temp": template.boltzmann_temp,
        "top_p": template.top_p,
        "policy": template.policy,
        "use_policy": template.use_policy,
        "aux_net": template.aux_net,
        "confidence": template.confidence,
    }
    values.update(overrides)
    clone = MortalEngine(template.brain, template.dqn, **values)
    clone.checkpoint = getattr(template, "checkpoint", None)
    return clone


class MortalControlPool:
    """One resolved checkpoint and one loaded template, many fresh arena seats."""

    def __init__(self, config: MortalControlConfig = MortalControlConfig()) -> None:
        from .evaluate import load_engine
        from .weights import resolve_mortal_checkpoint

        self.config = config
        self.checkpoint: ResolvedCheckpoint = resolve_mortal_checkpoint(
            config.weights, use_policy=bool(config.use_policy)
        )
        self._template = load_engine(
            self.checkpoint,
            device=config.device,
            use_policy=None,
            enable_quick_eval=bool(config.enable_quick_eval),
            enable_rule_based_agari_guard=bool(
                config.enable_rule_based_agari_guard
            ),
            boltzmann_epsilon=float(config.boltzmann_epsilon),
            boltzmann_temp=float(config.boltzmann_temp),
            name="control-template",
        )
        if config.use_policy and not self._template.use_policy:
            raise ValueError("configured checkpoint has no policy head")

    @property
    def identity(self) -> dict[str, str | bool]:
        return self.checkpoint.as_dict()

    def make_low_level(self, name: str) -> Any:
        return clone_mortal_engine(self._template, name)

    def make_arena(self, name: str, *, spectator: Any | None = None) -> Any:
        from .positions import MortalArenaEngine

        low_level = self.make_low_level(name)
        seat = MortalArenaEngine(name, low_level, spectator=spectator)
        seat.checkpoint = self.checkpoint
        return seat

    def make_table(self, names: Iterable[str]) -> list[Any]:
        table = [self.make_arena(str(name)) for name in names]
        if len(table) != 4:
            raise ValueError(f"expected four control seats, got {len(table)}")
        return table
