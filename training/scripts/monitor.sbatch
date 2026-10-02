#!/usr/bin/env bash
#SBATCH --job-name=Priming_OT_monitor
#SBATCH --partition=shared
#SBATCH --account=sham_lab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=2-00:00:00
set -euo pipefail
source "${OST_TRAIN_TASK:?}/scripts/common.sh"
exec "$PYTHON" "$TASK/scripts/status.py" --watch --interval 600
