#!/bin/bash
#SBATCH --job-name=jit_eval3seed
#SBATCH --partition=kempner_h100_priority
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --exclude=holygpu8a17501
#SBATCH --output=/n/home10/jaykim/woobin/OST/runs/logs/eval3seed-%j.out
set -euo pipefail
cd /n/home10/jaykim/woobin/OST
export PATH=/usr/bin:/bin
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME=/n/netscratch/sham_lab/Lab/workdir/huggingface
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/n/home10/jaykim/woobin/OST/tests/mmd/runs/torch_cache
export PYTHON=/n/home10/jaykim/.conda/envs/fdloss/bin/python
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 30000))
export SLURM_EXPORT_ENV=ALL
srun --kill-on-bad-exit=1 bash -c 'exec "$PYTHON" -m torch.distributed.run --nnodes=2 --nproc_per_node=4 --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" isolated_experiments/three_seed_eval_20260925/evaluate_v2.py --model jit_h --batch 64 --num-images 50000 --seeds "$1"' _ "$1"
