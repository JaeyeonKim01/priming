#!/bin/bash
#SBATCH --job-name=priming-world-eval
#SBATCH --account=kempner_sitanc_lab
#SBATCH --partition=kempner_h100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:00:00
set -euo pipefail
: "${WORLD_MODEL_PYTHON:?Set the environment python}"
: "${WORLD_MODEL_STORAGE:?Set the read-only asset root}"
: "${WORLD_MODEL_REFERENCE:?Set the extracted original source root}"
: "${WORLD_MODEL_RUN:?Set a dedicated reproduction output directory}"
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
"$WORLD_MODEL_PYTHON" -m world_modeling.scripts.reproduce_eval \
    --phase "${WORLD_MODEL_EVAL_PHASE:-subset}" \
    --output "$WORLD_MODEL_RUN" --storage "$WORLD_MODEL_STORAGE" \
    --reference "$WORLD_MODEL_REFERENCE"
