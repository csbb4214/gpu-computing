#!/bin/bash
#SBATCH --job-name=matrix_mul_benchmark_ifi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=10000
#SBATCH --partition=IFIAMD
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=matrix_mul_run.%j.out
#SBATCH --error=matrix_mul_run.%j.err

# Run all prebuilt matrix_mul binaries and collect results in CSV.
# No compilation happens here — assumes all binaries are already built
# e.g. by build_matrix_mul.sh in the same directory.

set -euo pipefail

echo "[INFO] Starting matrix_mul benchmark job..."
echo "[INFO] Host: $(hostname)"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Start time: $(date)"
echo

srun ./matrix_mul_N2048_float
