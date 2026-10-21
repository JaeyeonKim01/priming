#!/bin/bash
#SBATCH --job-name=world-train-pipeline
#SBATCH --account=kempner_sitanc_lab
#SBATCH --partition=kempner_h100
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:30:00
#SBATCH --mail-user=jaeyeon_kim@g.harvard.edu
#SBATCH --mail-type=END,FAIL
set -euo pipefail
: "${WORLD_MODEL_PYTHON:?Set the environment python}"
: "${WORLD_MODEL_RUN:?Prepare a dedicated run directory first}"
: "${WORLD_MODEL_STORAGE:?Set the read-only asset root}"
: "${DISTMATCH_SOURCE:?Set the recovered matcher source root}"
cd "$WORLD_MODEL_RUN/code"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export WORLD_MASTER_ADDR
WORLD_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
job_end=$(scontrol show job "$SLURM_JOB_ID" -o | sed -n 's/.* EndTime=\([^ ]*\).*/\1/p')
export WORLD_MODEL_DEADLINE
WORLD_MODEL_DEADLINE=$(date -d "$job_end" +%s)
run_stage() {
    if (( $(date +%s) >= WORLD_MODEL_DEADLINE - 300 )); then
        echo 'Near job deadline: retain completed checkpoints and resume in a new allocation.'
        exit 75
    fi
    srun --kill-on-bad-exit=1 bash -c '
        exec "$WORLD_MODEL_PYTHON" -m torch.distributed.run \
          --nnodes="$SLURM_JOB_NUM_NODES" --nproc_per_node=4 --node_rank="$SLURM_NODEID" \
          --master_addr="$WORLD_MASTER_ADDR" --master_port="$WORLD_MASTER_PORT" \
          -m "world_modeling.$1" "${@:2}"
    ' world-pipeline "$@"
}
advance() {
    local dataset="$1" stage="$2" target="$3"
    local directory="$WORLD_MODEL_RUN/$dataset/$stage"
    local latest="$directory/ckpt_save/ckpt-latest.pth"
    local completed=0
    local recovery=()
    if [[ -f "$latest" ]]; then
        completed=$("$WORLD_MODEL_PYTHON" -m world_modeling.scripts.training_pipeline step --checkpoint "$latest")
        if (( completed >= target )); then return; fi
        recovery=(--resume "$(readlink -f "$latest")")
    elif [[ "$stage" == posttrain ]]; then
        recovery=(--init-checkpoint "$WORLD_MODEL_RUN/$dataset/pretrain/ckpt_save/ckpt-step250.pth")
    fi
    local interval=100
    local settings=()
    if [[ "$stage" == posttrain ]]; then
        interval=50
        settings=(--set distmatch.verify_replay=true)
    fi
    run_stage "$stage" --config "world_modeling/configs/$stage/$dataset.yaml" \
        --output "$directory" --max-steps "$target" "${recovery[@]}" \
        --set dataloader.num_workers=4 --set "train.checkpoint_every=$interval" \
        "${settings[@]}"
}
for dataset in rt1 bridge; do
    advance "$dataset" pretrain 200
    advance "$dataset" pretrain 250
    advance "$dataset" posttrain 100
    advance "$dataset" posttrain 120
    videos=2
    if [[ "$dataset" == bridge ]]; then videos=3; fi
    run_stage eval --config "world_modeling/configs/eval/$dataset.yaml" \
        --output "$WORLD_MODEL_RUN/$dataset/eval" \
        --checkpoint "$WORLD_MODEL_RUN/$dataset/posttrain/ckpt_save/ckpt-step120.pth" \
        --max-videos-per-shard "$videos" --set dataloader.num_workers=2 --set evaluation.save_videos=0
done
"$WORLD_MODEL_PYTHON" -m world_modeling.scripts.training_pipeline summarize --run-root "$WORLD_MODEL_RUN"
