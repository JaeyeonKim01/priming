"""Evaluation-only distribution metrics.

Training objectives remain in :mod:`distmatch.objectives`.  This namespace is
deliberately separate so evaluation state and reference statistics never enter
the training matcher.
"""

from .frechet import (
    FrechetEvaluation,
    FrechetEvaluator,
    FrechetMoments,
    FrechetStats,
    frechet_distance,
    load_frechet_stats,
    save_frechet_stats,
)
from .inception import FIDInceptionExtractor, INCEPTION_WEIGHTS_URL

__all__ = [
    "FrechetEvaluation",
    "FrechetEvaluator",
    "FrechetMoments",
    "FrechetStats",
    "frechet_distance",
    "load_frechet_stats",
    "save_frechet_stats",
    "FIDInceptionExtractor",
    "INCEPTION_WEIGHTS_URL",
]
