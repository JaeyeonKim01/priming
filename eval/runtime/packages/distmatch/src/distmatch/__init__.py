from ._version import __version__
from .features import Feature
from .grad_caching import gradcache_backward
from .objectives.frechet import (
    EMAFrechet,
    FeatureStats,
    FrechetEvaluation,
    FrechetEvaluator,
    feature_stats,
)
from .objectives.forward_xm import ForwardXMResult, SharedForwardXM
from .objectives.mmd import MMD
from .objectives.sinkhorn_ot import (
    BalancedSinkhornOT,
    SinkhornOTResult,
    balanced_sinkhorn_plan,
)

__all__ = [
    "BalancedSinkhornOT",
    "EMAFrechet",
    "Feature",
    "FeatureStats",
    "FrechetEvaluation",
    "FrechetEvaluator",
    "ForwardXMResult",
    "MMD",
    "SharedForwardXM",
    "SinkhornOTResult",
    "__version__",
    "balanced_sinkhorn_plan",
    "feature_stats",
    "gradcache_backward",
]
