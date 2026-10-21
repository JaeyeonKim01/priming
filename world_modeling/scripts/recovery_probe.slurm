#!/bin/bash
#SBATCH --job-name=world-recovery-probe
#SBATCH --account=kempner_sitanc_lab
#SBATCH --partition=kempner_h100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=00:10:00
set -euo pipefail
: "${WORLD_MODEL_PYTHON:?Set environment python}"
: "${WORLD_MODEL_STORAGE:?Set read-only storage}"
: "${WORLD_MODEL_RUN:?Set the prepared snapshot root}"
: "${WORLD_MODEL_RECOVERY_RUN:?Set a fresh probe output directory}"
cd "$WORLD_MODEL_RUN/code"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
unset WORLD_MODEL_DEADLINE
mkdir -p "$WORLD_MODEL_RECOVERY_RUN"
# torchrun translates worker exit 75 into a launcher failure. Accept that only
# when the saved summary proves an intentional, coordinated checkpoint stop.
if "$WORLD_MODEL_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m world_modeling.pretrain --config world_modeling/configs/pretrain/rt1.yaml \
    --output "$WORLD_MODEL_RECOVERY_RUN" --max-steps 2 \
    --set dataloader.num_workers=2 --set train.max_session_seconds=0.001 \
    > "$WORLD_MODEL_RECOVERY_RUN/stop.log" 2>&1; then
    echo 'Expected an intentional checkpoint stop before the two-update target.'
    exit 1
fi
"$WORLD_MODEL_PYTHON" - <<'PY'
import json, os
from pathlib import Path
root=Path(os.environ['WORLD_MODEL_RECOVERY_RUN'])
summary=json.loads((root/'summary.json').read_text())
assert summary['stop_reason']=='session_time',summary
assert summary['completed_steps']==summary['last_saved_step']==1,summary
assert summary['target_steps']==2 and summary['world_size']==2,summary
assert (root/'ckpt_save/ckpt-step1.pth').is_file()
print('Coordinated stop saved update 1 on two ranks.',flush=True)
PY
"$WORLD_MODEL_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m world_modeling.pretrain --config world_modeling/configs/pretrain/rt1.yaml \
    --output "$WORLD_MODEL_RECOVERY_RUN" --max-steps 2 \
    --resume "$WORLD_MODEL_RECOVERY_RUN/ckpt_save/ckpt-step1.pth" \
    --set dataloader.num_workers=2 > "$WORLD_MODEL_RECOVERY_RUN/resume.log" 2>&1
"$WORLD_MODEL_PYTHON" - <<'PY'
import json, os, torch
from pathlib import Path
root=Path(os.environ['WORLD_MODEL_RECOVERY_RUN'])
summary=json.loads((root/'summary.json').read_text())
assert summary['initial_step']==1 and summary['completed_steps']==2,summary
checkpoint=torch.load(root/'ckpt_save/ckpt-step2.pth',map_location='cpu',mmap=True,weights_only=True)
assert len(checkpoint['rng_states'])==2
assert {int(x['step']) for x in checkpoint['optimizer']['state'].values()}=={2}
metrics=[json.loads(x) for x in (root/'metrics.jsonl').read_text().splitlines()]
assert [x['step'] for x in metrics]==[1,2]
result=dict(passed=True,world_size=2,intentional_stop_step=1,resumed_to_step=2,
            optimizer_steps=2,rng_states=2,slurm_job_id=os.environ['SLURM_JOB_ID'],summary=summary)
(root/'recovery-probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
PY
