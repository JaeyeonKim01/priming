#!/bin/bash
#SBATCH --job-name=world-production
#SBATCH --account=kempner_sitanc_lab
#SBATCH --partition=kempner_h100
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --mail-user=jaeyeon_kim@g.harvard.edu
#SBATCH --mail-type=END,FAIL
set -euo pipefail
: "${WORLD_MODEL_PYTHON:?Set the environment python}"
: "${WORLD_MODEL_RUN:?Prepare a dedicated production directory first}"
: "${WORLD_MODEL_STORAGE:?Set the read-only asset root}"
: "${DISTMATCH_SOURCE:?Set the recovered matcher source root}"
: "${WORLD_MODEL_DATASET:?Set rt1 or bridge}"
case "$WORLD_MODEL_DATASET" in
    rt1) pretrain_steps=84608 ;;
    bridge) pretrain_steps=53184 ;;
    *) echo 'Expected rt1 or bridge'; exit 2 ;;
esac
mkdir -p "$WORLD_MODEL_RUN/$WORLD_MODEL_DATASET"
exec 9>"$WORLD_MODEL_RUN/$WORLD_MODEL_DATASET/.allocation.lock"
flock -n 9 || { echo 'This task already has an active production launcher'; exit 1; }
cd "$WORLD_MODEL_RUN/code"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export WORLD_MASTER_ADDR
WORLD_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
job_end=$(scontrol show job "$SLURM_JOB_ID" -o | sed -n 's/.* EndTime=\([^ ]*\).*/\1/p')
export WORLD_MODEL_DEADLINE
WORLD_MODEL_DEADLINE=$(date -d "$job_end" +%s)
"$WORLD_MODEL_PYTHON" -m world_modeling.scripts.production verify --run-root "$WORLD_MODEL_RUN"
run_stage() {
    if (( $(date +%s) >= WORLD_MODEL_DEADLINE - 300 )); then
        echo 'Near allocation deadline; resume this frozen run in a new allocation.'
        exit 75
    fi
    srun --kill-on-bad-exit=1 bash -c '
        exec "$WORLD_MODEL_PYTHON" -m torch.distributed.run \
          --nnodes="$SLURM_JOB_NUM_NODES" --nproc_per_node=4 --node_rank="$SLURM_NODEID" \
          --master_addr="$WORLD_MASTER_ADDR" --master_port="$WORLD_MASTER_PORT" \
          -m "world_modeling.$1" "${@:2}"
    ' world-production "$@"
}
advance() {
    local stage="$1" target="$2"
    local latest="$WORLD_MODEL_RUN/$WORLD_MODEL_DATASET/$stage/ckpt_save/ckpt-latest.pth"
    local completed=0
    local recovery=()
    if [[ -f "$latest" ]]; then
        completed=$("$WORLD_MODEL_PYTHON" -m world_modeling.scripts.training_pipeline step --checkpoint "$latest")
        if (( completed > target )); then echo 'Checkpoint exceeds recipe target'; exit 1; fi
        recovery=(--resume "$(readlink -f "$latest")")
    fi
    if (( completed < target )); then
        run_stage "$stage" --config "$WORLD_MODEL_RUN/configs/$WORLD_MODEL_DATASET/$stage.yaml" "${recovery[@]}"
    fi
    "$WORLD_MODEL_PYTHON" -m world_modeling.scripts.production audit-stage \
        --run-root "$WORLD_MODEL_RUN" --dataset "$WORLD_MODEL_DATASET" --stage "$stage"
}
advance pretrain "$pretrain_steps"
advance posttrain 4000
run_stage eval --config "$WORLD_MODEL_RUN/configs/$WORLD_MODEL_DATASET/eval.yaml"
"$WORLD_MODEL_PYTHON" -m world_modeling.scripts.production summarize \
    --run-root "$WORLD_MODEL_RUN" --dataset "$WORLD_MODEL_DATASET"
