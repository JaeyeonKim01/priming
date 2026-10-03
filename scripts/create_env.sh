#!/usr/bin/env bash
# Run from any directory. CONDA_ENVS_PATH / CONDA_PKGS_DIRS may select scratch storage.
set -euo pipefail
export PYTHONNOUSERSITE=1
PRIMING_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PRIMING_ROOT"
CONDA_BIN=${CONDA_EXE:-conda}
# Match Conda's destination when callers choose scratch storage or another name.
TARGET=(-n priming)
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    case "${ARGS[i]}" in
        -p|--prefix|-n|--name)
            TARGET=("${ARGS[i]}" "${ARGS[i+1]:?Missing environment destination}") ;;
        --prefix=*) TARGET=(-p "${ARGS[i]#*=}") ;;
        --name=*) TARGET=(-n "${ARGS[i]#*=}") ;;
    esac
done
"$CONDA_BIN" env create --file environment.yml "$@"
"$CONDA_BIN" run --no-capture-output "${TARGET[@]}" python -m pip check
"$CONDA_BIN" run --no-capture-output "${TARGET[@]}" python scripts/check_environment.py
