#!/bin/bash
#SBATCH --job-name=priming_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=04:00:00
set -euo pipefail
cd "${PRIMING_ROOT:-${SLURM_SUBMIT_DIR:?}}"
test -f eval/evaluate.py
: "${PRIMING_EVAL_OUTPUT:?Set PRIMING_EVAL_OUTPUT to a fresh scratch directory}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME="${HF_HOME:-$PWD/assets/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME="${TORCH_HOME:-$PWD/assets/torch}"
export DRIFT_COMPILE=0
PYTHON="${PRIMING_PYTHON:-$(command -v python)}"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  eval/evaluate.py --output-dir "$PRIMING_EVAL_OUTPUT" "$@"
