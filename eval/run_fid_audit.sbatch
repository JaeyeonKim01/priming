#!/bin/bash
#SBATCH --job-name=ost_public_fid_audit
#SBATCH --partition=shared
#SBATCH --account=sham_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${OST_REPO_ROOT:-${SLURM_SUBMIT_DIR:?}}"
: "${OST_EVAL_OUTPUT:?Set OST_EVAL_OUTPUT to the new scratch batch}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /n/home10/jaykim/.conda/envs/fdloss/bin/python \
  OST_public/eval/audit_saved_moments.py --output-dir "$OST_EVAL_OUTPUT" "$@"
