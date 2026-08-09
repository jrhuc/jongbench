from verifiers.v1.harnesses.null import NullHarness

from riichi_decision_v1.taskset import (
    SAMPLE_BANK,
    RiichiDecisionConfig,
    RiichiDecisionData,
    RiichiDecisionTask,
    RiichiDecisionTaskset,
)

# Re-exporting a Harness subclass makes it the taskset's default harness, so
# `eval riichi_decision_v1` runs the plain chat loop instead of a bash agent.
__all__ = [
    "SAMPLE_BANK",
    "NullHarness",
    "RiichiDecisionConfig",
    "RiichiDecisionData",
    "RiichiDecisionTask",
    "RiichiDecisionTaskset",
]
