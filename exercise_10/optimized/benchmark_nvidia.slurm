#!/bin/bash
#SBATCH --job-name=matrix_mul_benchmark
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=IFIgpu2070
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --error=matrix_mul_%j.err
#SBATCH --output=matrix_mul_%j.out

set -euo pipefail

echo "[INFO] Starting matrix multiplication benchmark"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Start time: $(date)"
echo

# ---------------- CONFIG ----------------
RESULTS="matrix_mul_results_nvidia_opt.csv"
MATRIX_SIZES=(512 1024 2000 2048)
RUNS=3
PRECISIONS=(float double)
# ----------------------------------------

rm -f "$RESULTS" missing_binaries.log

# CSV Header (matches program output!)
echo "precision,N,time_ms" >"$RESULTS"

for N in "${MATRIX_SIZES[@]}"; do
  for PREC in "${PRECISIONS[@]}"; do

    BIN="./matrix_mul_N${N}_${PREC}"

    echo "[INFO] Running N=$N precision=$PREC ($RUNS runs)"

    if [[ ! -x "$BIN" ]]; then
      echo "[WARNING] Missing binary: $BIN" | tee -a missing_binaries.log
      continue
    fi

    for ((run = 1; run <= RUNS; run++)); do
      echo "[INFO]   Run $run"
      "$BIN" >>"$RESULTS"
    done

  done
done

echo
echo "[DONE] Results written to $RESULTS"

if [[ -f missing_binaries.log ]]; then
  echo "[NOTE] Some binaries were missing — see missing_binaries.log"
fi

echo "[INFO] End time: $(date)"
