#!/usr/bin/env bash
#SBATCH --job-name=OST_B_e160
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_sitanc_lab
#SBATCH --nodes=4
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/n/home10/jaykim/woobin/OST/runs/logs/%x-%j.out
set -euo pipefail
source /n/home10/jaykim/woobin/OST/isolated_experiments/ost_b_pretrain_pair_20260912/scripts/common.sh
: "${ARM:?ARM is required}"
case "$ARM" in perceptual|no_perceptual) ;; *) exit 2 ;; esac
"$PYTHON" "$STUDY/scripts/production_gate.py" --verify
export RUN_DIR="/n/home10/jaykim/woobin/OST/fdloss_work_dirs/OST/OST_B_t08_e160_${ARM}_seed1_20260912"
export JOB_DIR="$STUDY/jobs/production_${ARM}_${SLURM_JOB_ID}"
JOB_STORE="/n/netscratch/sham_lab/Lab/jaykim/ost_b_pretrain_pair_20260912/production_${ARM}_${SLURM_JOB_ID}"
mkdir -p "$JOB_STORE"
if [[ ! -e "$JOB_DIR" ]]; then ln -s "$JOB_STORE" "$JOB_DIR"; fi
if [[ ${SLURM_RESTART_COUNT:-0} == 0 ]]; then
    test ! -e "$RUN_DIR"
else
    test -f "$RUN_DIR/checkpoints/latest.pth"
fi
export PREEMPT_FLAG_FILE="$JOB_DIR/preempt_${SLURM_RESTART_COUNT:-0}.flag"
export PAIR_CACHE_ROOT="${TMPDIR:-/tmp}/ost_b_pair_${SLURM_JOB_ID}/compiler_cache"
trap 'touch "$PREEMPT_FLAG_FILE"' USR1
cd "$STUDY/runtime"
if [[ ! -f "$RUN_DIR/training_complete.json" ]]; then
    run_ranks scripts/pair_rank_entry.py --config "$STUDY/configs/$ARM.yaml" \
        --output "$RUN_DIR" --manifest "$STUDY/reports/manifest.json" &
    worker_pid=$!
    set +e
    wait_for_ranks "$worker_pid"
    worker_status=$?
    set -e
    if [[ -f "$PREEMPT_FLAG_FILE" ]]; then
        "$PYTHON" "$STUDY/scripts/check_completion.py" --run "$RUN_DIR" --preempt
        if [[ ${SLURM_RESTART_COUNT:-0} -ge 2 ]]; then
            echo 'Two recovery attempts exhausted; complete checkpoint preserved.'
            exit 75
        fi
        scontrol requeue "$SLURM_JOB_ID"
        exit 0
    fi
    if [[ $worker_status != 0 ]]; then exit "$worker_status"; fi
    "$PYTHON" "$STUDY/scripts/check_completion.py" --run "$RUN_DIR"
fi
# Evaluation stays in this arm's eight-GPU allocation and uses the frozen L evaluator.
cd "$STUDY/evaluation_runtime"
run_ranks "$STUDY/scripts/evaluate_terminal.py" --config "$STUDY/configs/evaluation.json" \
    "checkpointing.resume_from=$RUN_DIR/checkpoints/step_0200160.pth" \
    "logging.output_dir=$RUN_DIR" logging.project=terminal_eval logging.exp_name=frozen_50k &
eval_pid=$!
set +e
wait_for_ranks "$eval_pid"
eval_status=$?
set -e
if [[ $eval_status != 0 && -f "$PREEMPT_FLAG_FILE" && ${SLURM_RESTART_COUNT:-0} -lt 2 ]]; then
    test -f "$RUN_DIR/training_complete.json"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$eval_status"
