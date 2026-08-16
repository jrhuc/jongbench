"""Typed records and reducers for causal jongbench experiments.

Runtime intervention helpers live in :mod:`jongbench.experiments.branch` and
:mod:`jongbench.experiments.matched_control`. They are not imported here, keeping
artifact inspection and statistical reduction torch-free.
"""

from .capsule import ReplayCapsule, ScriptedDecision
from .observations import (
    AllControlBaseline,
    BranchResult,
    MatchedControlResult,
    PairedArmObservation,
)
from .reduce import (
    BranchReduction,
    ClusterEstimate,
    MatchedControlReduction,
    PairedArmReduction,
    reduce_factual_branches,
    reduce_matched_controls,
    reduce_paired_arms,
)

__all__ = [
    "AllControlBaseline",
    "BranchReduction",
    "BranchResult",
    "ClusterEstimate",
    "MatchedControlReduction",
    "MatchedControlResult",
    "PairedArmObservation",
    "PairedArmReduction",
    "ReplayCapsule",
    "ScriptedDecision",
    "reduce_factual_branches",
    "reduce_matched_controls",
    "reduce_paired_arms",
]
