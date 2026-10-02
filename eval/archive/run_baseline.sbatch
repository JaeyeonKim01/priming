#!/bin/bash
#SBATCH --job-name=eval3seed
#SBATCH --partition=kempner_h100_priority
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/n/home10/jaykim/woobin/OST/runs/logs/eval3seed-%j.out
set -euo pipefail
cd /n/home10/jaykim/woobin/OST
export PATH=/usr/bin:/bin
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME=/n/netscratch/sham_lab/Lab/workdir/huggingface
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/n/home10/jaykim/woobin/OST/tests/mmd/runs/torch_cache
export DRIFT_COMPILE=0
PYTHON=/n/home10/jaykim/.conda/envs/fdloss/bin/python
MODEL="$1"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 isolated_experiments/three_seed_eval_20260925/evaluate_v2.py --model "$MODEL" --num-images 1000 --seeds 1
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 isolated_experiments/three_seed_eval_20260925/evaluate_v2.py --model "$MODEL" --num-images 50000 --seeds 1 2 3
