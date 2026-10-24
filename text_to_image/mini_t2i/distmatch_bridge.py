"""Resolve the external DistMatch source selected for T2I post-training."""

import os
from pathlib import Path
import sys

configured = os.environ.get("DISTMATCH_SOURCE")
source = None
if configured:
    root = Path(configured).expanduser().resolve()
    if (root / "src" / "distmatch" / "__init__.py").is_file():
        source = root / "src"
    elif (root / "distmatch" / "__init__.py").is_file():
        source = root
    elif root.name == "distmatch" and (root / "__init__.py").is_file():
        source = root.parent
    else:
        raise FileNotFoundError(f"DISTMATCH_SOURCE does not contain a DistMatch package: {root}")
    sys.path.insert(0, str(source))

import distmatch

if source is not None and not Path(distmatch.__file__).resolve().is_relative_to(source):
    raise ImportError("a different distmatch is already imported; start a fresh process with DISTMATCH_SOURCE")

from distmatch import DistMatch, Feature, FeatureBranches, MMD, Sinkhorn

__all__ = ["DistMatch", "Feature", "FeatureBranches", "MMD", "Sinkhorn"]
