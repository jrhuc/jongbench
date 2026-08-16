"""Public reducer facade.

Execution layers emit facts. These reducers own estimands, completeness checks, and
uncertainty so a UI row average cannot silently redefine a benchmark.
"""

from .reduce_branches import (
    BranchDecisionEstimate,
    BranchReduction,
    reduce_factual_branches,
)
from .reduce_controls import MatchedControlReduction, reduce_matched_controls
from .reduce_paired import PairedArmReduction, reduce_paired_arms
from .statistics import ClusterEstimate, cluster_bootstrap_mean

__all__ = [
    "BranchDecisionEstimate",
    "BranchReduction",
    "ClusterEstimate",
    "MatchedControlReduction",
    "PairedArmReduction",
    "cluster_bootstrap_mean",
    "reduce_factual_branches",
    "reduce_matched_controls",
    "reduce_paired_arms",
]
