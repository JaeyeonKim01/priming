#!/usr/bin/env bash
#SBATCH --job-name=OST_L_pretrain_t08
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_sitanc_lab
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=jason1214@snu.ac.kr
#SBATCH --output=runs/%x-%j.out

set -euo pipefail

# Runtime settings.
: "${REPO_DIR:=/n/home10/jaykim/woobin/OST}"
: "${PYTHON:=/n/home10/jaykim/.conda/envs/fdloss/bin/python}"
: "${CONFIG:=configs/train_ost.yaml}"
: "${OUTPUT_DIR:=./fdloss_work_dirs}"
: "${PROJECT:=OST}"
: "${EXP_NAME:=OST_L_aux_unif_t08_${SLURM_JOB_ID:-local}}"

cd "${REPO_DIR}"
mkdir -p runs/logs

export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
export HF_HOME=/n/netscratch/sham_lab/Lab/workdir/huggingface
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# Re-enter this script before Slurm's time limit. Training auto-resumes from
# the latest checkpoint already present in the experiment directory.
trap 'echo "[$(date)] time limit approaching; requeueing job $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1

run_distributed() {
    srun --kill-on-bad-exit=1 \
        bash -c '
            python=$1
            shift
            exec "$python" -m torch.distributed.run \
                --nnodes="$SLURM_JOB_NUM_NODES" \
                --nproc_per_node="$SLURM_GPUS_ON_NODE" \
                --node_rank="$SLURM_NODEID" \
                --master_addr="$MASTER_ADDR" \
                --master_port="$MASTER_PORT" \
                "$@"
        ' bash "$@" &
    wait $!
}

echo "job_id=${SLURM_JOB_ID:-none} host=$(hostname)"
echo "nodes=${SLURM_JOB_NUM_NODES} gpus_per_node=${SLURM_GPUS_ON_NODE}"
echo "config=${CONFIG}"

run_distributed "${PYTHON}" \
    scripts/train_ost.py \
    --config "${CONFIG}" \
    model.model=OST_L model.t_dist=uniform model.t_min=0.8 model.t_max=0.8 \
    train.epochs=160 train.micro_batch_size=64 train.gradient_accumulation_steps=2 \
    train.output_dir="${OUTPUT_DIR}" train.project="${PROJECT}" train.exp_name="${EXP_NAME}"

echo "OST pretraining complete."
