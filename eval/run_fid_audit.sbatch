#!/bin/bash
#SBATCH --job-name=priming_fid_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${PRIMING_ROOT:-${SLURM_SUBMIT_DIR:?}}"
: "${PRIMING_EVAL_OUTPUT:?Set PRIMING_EVAL_OUTPUT to the new scratch batch}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec "${PRIMING_PYTHON:-python}" \
  eval/audit_saved_moments.py --output-dir "$PRIMING_EVAL_OUTPUT" "$@"
