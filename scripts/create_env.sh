#!/usr/bin/env bash
# Run from any directory. CONDA_ENVS_PATH / CONDA_PKGS_DIRS may select scratch storage.
set -euo pipefail
export PYTHONNOUSERSITE=1
PRIMING_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PRIMING_ROOT"
CONDA_BIN=${CONDA_EXE:-conda}
"$CONDA_BIN" env create --file environment.yml "$@"
"$CONDA_BIN" run --no-capture-output -n priming python -m pip check
"$CONDA_BIN" run --no-capture-output -n priming python scripts/check_environment.py
