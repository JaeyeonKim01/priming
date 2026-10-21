#!/bin/bash
#SBATCH --job-name=priming-world-8rank
#SBATCH --account=kempner_sitanc_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu32_id50
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00
set -euo pipefail
: "${WORLD_MODEL_PYTHON:?Set WORLD_MODEL_PYTHON}"
: "${WORLD_MODEL_STORAGE:?Set WORLD_MODEL_STORAGE}"
: "${DISTMATCH_SOURCE:?Set DISTMATCH_SOURCE}"
: "${WORLD_MODEL_RUN:?Set WORLD_MODEL_RUN to a fresh output directory}"
cd "${SLURM_SUBMIT_DIR}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
master_host=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_MASTER_ADDR="$master_host"
export WORLD_MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
run_stage() {
    srun --kill-on-bad-exit=1 bash -c '
        exec "$WORLD_MODEL_PYTHON" -m torch.distributed.run \
          --nnodes="$SLURM_JOB_NUM_NODES" --nproc_per_node=4 --node_rank="$SLURM_NODEID" \
          --master_addr="$WORLD_MASTER_ADDR" --master_port="$WORLD_MASTER_PORT" \
          -m "world_modeling.$1" "${@:2}"
    ' world-smoke "$@"
}
for dataset in rt1 bridge; do
    pre="$WORLD_MODEL_RUN/$dataset/pretrain"
    post="$WORLD_MODEL_RUN/$dataset/posttrain"
    run_stage pretrain --config "world_modeling/configs/pretrain/$dataset.yaml" --output "$pre" --max-steps 1 --set dataloader.num_workers=2
    run_stage posttrain --config "world_modeling/configs/posttrain/$dataset.yaml" --output "$post" --init-checkpoint "$pre/ckpt_save/ckpt-step1.pth" --max-steps 1 --set dataloader.num_workers=2 --set distmatch.verify_replay=true
done
