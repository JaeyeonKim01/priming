#!/usr/bin/env bash
set -euo pipefail
overrides=()
if [[ "${MODE:-probe}" == train && "${SLURM_RESTART_COUNT:-0}" != 0 ]]; then
  test -f "$RUN_DIR/checkpoints/latest.pth"
  overrides+=("checkpointing.resume_from=$RUN_DIR/checkpoints/latest.pth" checkpointing.auto_resume=false)
fi
exec "$PYTHON" -m torch.distributed.run \
  --nnodes="$SLURM_JOB_NUM_NODES" --nproc-per-node="$SLURM_GPUS_ON_NODE" --node-rank="$SLURM_NODEID" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  "$JOB_DIR/scripts/rank_entry.py" --config "$JOB_DIR/config.json" "${overrides[@]}"
