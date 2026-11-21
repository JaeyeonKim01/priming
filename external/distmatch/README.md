# DistMatch 0.6.0

This is the DistMatch package used by the MiniT2I one-step post-training code.
The Python package files in `src/distmatch/` were copied unchanged from the
local `~/distmatch/src/distmatch/` checkout at version 0.6.0. The package
provides `DistMatch`, `Feature`, `FeatureBranches`, `Sinkhorn`, and `MMD`.

The T2I bridge loads this copy automatically from a Priming checkout. The world
modeling config also defaults to this directory. To use another compatible
source, set `DISTMATCH_SOURCE` to its repository root, `src/` directory, or
package directory. To install the bundled package separately, run:

```bash
python -m pip install -e external/distmatch
```

The core package requires PyTorch 2.1 or newer. Optional evaluation helpers
have extra dependencies listed in `pyproject.toml`.

## License

This bundled package is licensed under Apache License 2.0, the same license as
the Priming repository. See [LICENSE](LICENSE).
