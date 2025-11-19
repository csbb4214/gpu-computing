#!/bin/bash
#SBATCH --job-name=reduction_benchmark_ifi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=10000
#SBATCH --partition=IFIgpu2070
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=reduction_run.%j.out
#SBATCH --error=reduction_run.%j.err

# Run all prebuilt reduction binaries and collect results in CSV.
# No compilation happens here — assumes all binaries are already built
# by build.sh in the same directory.

set -euo pipefail

echo "[INFO] Starting reduction benchmark job..."
echo "[INFO] Host: $(hostname)"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Start time: $(date)"
echo

# --- config ---
RESULTS="results.csv"
N_VALUES=(1024 1048576 536870912)
RUNS=10

# --- cleanup old result / log files ---
rm -f "$RESULTS" 2>/dev/null || true
rm -f missing_binaries.log 2>/dev/null || true

# --- CSV header (wie im Original-Batch-Skript) ---
echo "version,precision,N,result,elapsed_ms" >"$RESULTS"

# --- run loop ---
for N in "${N_VALUES[@]}"; do
  echo "[INFO] N=$N ($RUNS runs each binary)"

  #################
  # --- int ---
  #################
  echo "[INFO] Running int binaries for N=$N"

  for ((x = 1; x <= RUNS; x++)); do
    # sequential int
    if [[ -x "./sequential_reduction_N${N}" ]]; then
      ./sequential_reduction_N${N} >>"$RESULTS"
    else
      echo "[WARNING] Missing binary: ./sequential_reduction_N${N}" | tee -a missing_binaries.log
    fi

    # parallel int
    if [[ -x "./parallel_reduction_N${N}" ]]; then
      ./parallel_reduction_N${N} >>"$RESULTS"
    else
      echo "[WARNING] Missing binary: ./parallel_reduction_N${N}" | tee -a missing_binaries.log
    fi

    # multistage int
    if [[ -x "./multistage_reduction_N${N}" ]]; then
      ./multistage_reduction_N${N} >>"$RESULTS"
    else
      echo "[WARNING] Missing binary: ./multistage_reduction_N${N}" | tee -a missing_binaries.log
    fi
  done

  ###################
  # --- float ---
  ###################
  echo "[INFO] Running float binaries for N=$N"

  for ((x = 1; x <= RUNS; x++)); do
    # sequential float
    if [[ -x "./sequential_reduction_N${N}_float" ]]; then
      ./sequential_reduction_N${N}_float >>"$RESULTS"
    else
      echo "[WARNING] Missing binary: ./sequential_reduction_N${N}_float" | tee -a missing_binaries.log
    fi

    # parallel float
    if [[ -x "./parallel_reduction_N${N}_float" ]]; then
      ./parallel_reduction_N${N}_float >>"$RESULTS"
    else
      echo "[WARNING] Missing binary: ./parallel_reduction_N${N}_float" | tee -a missing_binaries.log
    fi

    # multistage float
    if [[ -x "./multistage_reduction_N${N}_float" ]]; then
      ./multistage_reduction_N${N}_float >>"$RESULTS"
    else
      echo "[WARNING] Missing binary: ./multistage_reduction_N${N}_float" | tee -a missing_binaries.log
    fi
  done
done

echo
echo "[DONE] Results written to ${RESULTS}"
if [[ -f missing_binaries.log ]]; then
  echo "[NOTE] Some binaries were missing — see missing_binaries.log"
fi

echo "[INFO] End time: $(date)"

