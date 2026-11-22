#!/bin/bash
#SBATCH --job-name=decompile-eval
#SBATCH --account=PAS3239
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/3.12
source .venv/bin/activate
python -m harness run "$@" --wait
