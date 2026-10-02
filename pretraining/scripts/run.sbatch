#!/usr/bin/env bash
# B: sbatch --nodes=4 --gpus-per-node=2 run.sbatch /absolute/bundle
# L: sbatch --nodes=2 --gpus-per-node=4 run.sbatch /absolute/bundle
#SBATCH --job-name=priming_pretrain
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_sitanc_lab
#SBATCH --nodes=4
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --open-mode=append
#SBATCH --output=priming-pretrain-%j.out
set -euo pipefail
: "${PRETRAINING_ROOT:?Set PRETRAINING_ROOT to the absolute OST_public/pretraining path}"
: "${PYTHON:=python}"
export PYTHON PRETRAINING_ROOT
export BUNDLE
BUNDLE=$(realpath "${1:?Pass the prepared bundle directory}")
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((23000 + SLURM_JOB_ID % 15000))
export PREEMPT_FLAG_FILE="$BUNDLE/preempt_${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}.flag"
test ! -e "$PREEMPT_FLAG_FILE"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
trap 'touch "$PREEMPT_FLAG_FILE"' USR1
srun --kill-on-bad-exit=1 --export=ALL bash -c '
    args=(--bundle "$BUNDLE")
    if [[ ${RESUME:-0} == 1 ]]; then args+=(--resume); fi
    exec "$PYTHON" -m torch.distributed.run \
        --nnodes="$SLURM_JOB_NUM_NODES" --nproc_per_node="$SLURM_GPUS_ON_NODE" \
        --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
        "$PRETRAINING_ROOT/scripts/train.py" "${args[@]}"
' &
worker=$!
set +e
while true; do
    wait "$worker"
    code=$?
    if kill -0 "$worker" 2>/dev/null; then continue; fi
    break
done
set -e
# torchrun maps the B trainer's exit 75 to its own nonzero status.
if [[ -f "$PREEMPT_FLAG_FILE" ]]; then code=75; fi
exit "$code"
