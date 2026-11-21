"""Sample-in, loss-out distribution matching. Sampling belongs to the caller."""

from .features import Feature, FeatureBranches
from .matching import DistMatch
from .objectives import MMD, Sinkhorn, SinkhornConvergenceError

__all__ = ["DistMatch", "Feature", "FeatureBranches", "MMD", "Sinkhorn", "SinkhornConvergenceError"]
__version__ = "0.6.0"
