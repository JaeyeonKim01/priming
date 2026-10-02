#!/bin/bash
#SBATCH --job-name=ost_public_eval
#SBATCH --partition=kempner_h100_priority
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=04:00:00
set -euo pipefail
cd "${OST_REPO_ROOT:-${SLURM_SUBMIT_DIR:?}}"
test -f OST_public/eval/evaluate.py
: "${OST_EVAL_OUTPUT:?Set OST_EVAL_OUTPUT to a fresh scratch directory}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME="${OST_HF_HOME:-/n/netscratch/sham_lab/Lab/workdir/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME="${OST_TORCH_HOME:-$PWD/tests/mmd/runs/torch_cache}"
export DRIFT_COMPILE=0
PYTHON="${OST_PYTHON:-/n/home10/jaykim/.conda/envs/fdloss/bin/python}"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  OST_public/eval/evaluate.py --output-dir "$OST_EVAL_OUTPUT" "$@"
