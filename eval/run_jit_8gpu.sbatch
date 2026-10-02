#!/bin/bash
#SBATCH --job-name=priming_jit_eval
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
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
export PRIMING_PYTHON="${PRIMING_PYTHON:-$(command -v python)}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT="$((15000 + SLURM_JOB_ID % 30000))"
export SLURM_EXPORT_ENV=ALL
srun --kill-on-bad-exit=1 bash -c 'exec "$PRIMING_PYTHON" -m torch.distributed.run --nnodes="$SLURM_JOB_NUM_NODES" --nproc_per_node="$SLURM_GPUS_ON_NODE" --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" eval/evaluate.py --output-dir "$PRIMING_EVAL_OUTPUT" --model jit_h --batch 64 "$@"' _ "$@"
