#!/usr/bin/env bash
#SBATCH --job-name=Priming_OT
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_sitanc_lab
#SBATCH --nodes=4
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@3600
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
source "${OST_TRAIN_TASK:?}/scripts/common.sh"
export ARM=${1:?b160 or l160} MODE=${2:?probe or train}
[[ "$ARM" == b160 || "$ARM" == l160 ]]
[[ "$MODE" == probe || "$MODE" == train ]]
[[ "${SLURM_RESTART_COUNT:-0}" -le 2 ]]
export JOB_DIR="$SCRATCH_ROOT/${MODE}_${ARM}_${SLURM_JOB_ID}"
export RUN_NAME RUN_DIR
RUN_NAME=$("$PYTHON" -c 'import json,os; c=json.load(open(os.environ["TASK"]+"/configs/"+os.environ["ARM"]+".json")); print(c["logging"]["exp_name"])')
if [[ "$MODE" == probe ]]; then RUN_NAME="${RUN_NAME}_probe_${SLURM_JOB_ID}"; fi
RUN_DIR=$("$PYTHON" -c 'import json,os; from pathlib import Path; c=json.load(open(os.environ["TASK"]+"/configs/"+os.environ["ARM"]+".json")); print(Path(c["logging"]["output_dir"])/c["logging"]["project"]/os.environ["RUN_NAME"])')
if [[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]; then
  test ! -e "$RUN_DIR"
  if [[ "$MODE" == train ]]; then "$PYTHON" "$TASK/scripts/verify.py" --gate "$ARM"; fi
  "$PYTHON" "$TASK/scripts/snapshot.py" --job "$JOB_DIR"
  "$PYTHON" "$JOB_DIR/scripts/render_config.py" --task "$JOB_DIR" --arm "$ARM" \
    --mode "$MODE" --run-name "$RUN_NAME" --out "$JOB_DIR/config.json"
else
  [[ "$MODE" == train ]]
  test -f "$RUN_DIR/checkpoints/latest.pth"
fi
export PYTHONPATH="$JOB_DIR/runtime:$JOB_DIR/runtime/packages/distmatch/src:$JOB_DIR/scripts"
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
cd "$JOB_DIR/runtime"
if [[ "$MODE" == probe ]]; then
  cp "$JOB_DIR/config.json" "$JOB_DIR/reports/initial_config.json"
  srun --export=ALL --kill-on-bad-exit=1 /bin/bash "$JOB_DIR/scripts/worker.sh"
  "$PYTHON" "$JOB_DIR/scripts/audit_checkpoint.py" --config "$JOB_DIR/config.json" --output "$JOB_DIR/reports/initial_audit.json"
  "$PYTHON" "$JOB_DIR/scripts/render_config.py" --task "$JOB_DIR" --arm "$ARM" \
    --mode probe --updates 135 --resume "$RUN_DIR/checkpoints/step_0000129.pth" \
    --run-name "$RUN_NAME" --out "$JOB_DIR/config.json"
  export RANK_MODE=restore
  srun --export=ALL --kill-on-bad-exit=1 /bin/bash "$JOB_DIR/scripts/worker.sh"
  unset RANK_MODE
  srun --export=ALL --kill-on-bad-exit=1 /bin/bash "$JOB_DIR/scripts/worker.sh"
  "$PYTHON" "$JOB_DIR/scripts/audit_checkpoint.py" --config "$JOB_DIR/config.json" --output "$JOB_DIR/reports/resume_audit.json"
  cp "$RUN_DIR/resume_state_audit.json" "$RUN_DIR/training_flops.json" "$RUN_DIR/memory_audit.json" "$JOB_DIR/reports/"
  "$PYTHON" "$JOB_DIR/scripts/finish_preflight.py" --job "$JOB_DIR" --task "$TASK" --arm "$ARM"
  exit 0
fi
export PREEMPT_FLAG_FILE="$JOB_DIR/preempt_${SLURM_RESTART_COUNT:-0}.flag"
trap 'touch "$PREEMPT_FLAG_FILE"' USR1
srun --export=ALL --kill-on-bad-exit=1 /bin/bash "$JOB_DIR/scripts/worker.sh" &
worker_pid=$!
set +e
wait "$worker_pid"
worker_status=$?
if [[ -f "$PREEMPT_FLAG_FILE" ]] || kill -0 "$worker_pid" 2>/dev/null; then
  wait "$worker_pid"
  worker_status=$?
fi
set -e
if [[ "$worker_status" != 0 ]]; then exit "$worker_status"; fi
if [[ -f "$PREEMPT_FLAG_FILE" ]]; then
  test -f "$RUN_DIR/checkpoints/latest.pth"
  [[ "${SLURM_RESTART_COUNT:-0}" -lt 2 ]]
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
"$PYTHON" "$JOB_DIR/scripts/audit_evaluations.py" --csv "$RUN_DIR/final_eval_summary.csv" --output "$JOB_DIR/reports/evaluation_series_audit.json"
"$PYTHON" "$JOB_DIR/scripts/audit_checkpoint.py" --config "$JOB_DIR/config.json" --output "$JOB_DIR/reports/completion_audit.json"
cp "$JOB_DIR/reports/completion_audit.json" "$TASK/reports/completion_${ARM}.json"
