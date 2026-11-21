"""Load the bundled DistMatch package, with an optional source override."""

import os
from pathlib import Path
import sys

configured = os.environ.get("DISTMATCH_SOURCE")
root = (Path(configured).expanduser().resolve() if configured else
        Path(__file__).resolve().parents[2] / "external" / "distmatch")
if (root / "src" / "distmatch" / "__init__.py").is_file():
    source = root / "src"
elif (root / "distmatch" / "__init__.py").is_file():
    source = root
elif root.name == "distmatch" and (root / "__init__.py").is_file():
    source = root.parent
else:
    raise FileNotFoundError(f"DistMatch source does not contain a package: {root}")
sys.path.insert(0, str(source))

import distmatch

if not Path(distmatch.__file__).resolve().is_relative_to(source):
    raise ImportError("a different distmatch is already imported; start a fresh process with DISTMATCH_SOURCE")

from distmatch import DistMatch, Feature, FeatureBranches, MMD, Sinkhorn

__all__ = ["DistMatch", "Feature", "FeatureBranches", "MMD", "Sinkhorn"]
