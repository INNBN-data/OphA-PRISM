#!/bin/bash
#SBATCH --job-name=ophaprism_ref
#SBATCH --array=0-49%16
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=ophaprism_%A_%a.log
set -euo pipefail
# Usage from repository root: sbatch scripts/reference_fold.slurm /server/project VISION run_name
if [ "$#" -ne 3 ]; then
  echo "Usage: sbatch scripts/reference_fold.slurm DATA_ROOT TASK RUN_NAME" >&2
  exit 2
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
repeat_index=$((SLURM_ARRAY_TASK_ID / 5))
fold_index=$((SLURM_ARRAY_TASK_ID % 5))
python3 run_pipeline.py train --data-root "$1" --task "$2" --run "$3" --repeat "$repeat_index" --fold "$fold_index" --resume
