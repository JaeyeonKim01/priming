#!/usr/bin/env bash
export TASK=${OST_TRAIN_TASK:?Set OST_TRAIN_TASK to the training directory}
export PYTHON=${OST_TRAIN_PYTHON:-$(command -v python)}
export PRIMING_ROOT=${PRIMING_ROOT:-$(cd "$TASK/.." && pwd)}
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export SCRATCH_ROOT HF_HOME TORCH_HOME
SCRATCH_ROOT=$("$PYTHON" "$TASK/scripts/paths.py" --site-key scratch_root)
HF_HOME=$("$PYTHON" "$TASK/scripts/paths.py" --site-key hf_home)
TORCH_HOME=$("$PYTHON" "$TASK/scripts/paths.py" --site-key torch_home)
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_COMPILE_THREADS=4
