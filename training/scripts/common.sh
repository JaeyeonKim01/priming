#!/usr/bin/env bash
export TASK=${OST_TRAIN_TASK:?Set OST_TRAIN_TASK to the training directory}
export PYTHON=${OST_TRAIN_PYTHON:-/n/home10/jaykim/.conda/envs/fdloss/bin/python}
export PATH="$(dirname "$PYTHON"):/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export SCRATCH_ROOT
SCRATCH_ROOT=$("$PYTHON" -c 'import json,os; print(json.load(open(os.environ["TASK"]+"/configs/site.json"))["scratch_root"])')
export HF_HOME
HF_HOME=$("$PYTHON" -c 'import json,os; print(json.load(open(os.environ["TASK"]+"/configs/site.json"))["hf_home"])')
export TORCH_HOME
TORCH_HOME=$("$PYTHON" -c 'import json,os; print(json.load(open(os.environ["TASK"]+"/configs/site.json"))["torch_home"])')
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_COMPILE_THREADS=4
