#!/bin/bash
#SBATCH --job-name=jacobi_benchmark_ifi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=10000
#SBATCH --partition=IFIgpu2070
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=jacobi_run_only.%j.out
#SBATCH --error=jacobi_run_only.%j.err

# Run all prebuilt Jacobi binaries and collect results in CSV.
# No compilation happens here — assumes all binaries are already built.

set -euo pipefail

echo "[INFO] Starting Jacobi run job..."
echo "[INFO] Host: $(hostname)"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Start time: $(date)"
echo

# --- config ---
RESULTS="results.csv"
N_VALUES=(1024 2048)
IT_VALUES=(10 100 1000)
RUNS=10

# --- cleanup ---
rm -f "$RESULTS" 2>/dev/null || true
rm -f kernel_times_*.csv 2>/dev/null || true

# --- CSV header (updated to match local script) ---
echo "precision,N,IT,total_kernel,total_read,total_write,write_f,write_tmp,write_u,average_queue" >"$RESULTS"

# --- run loop ---
for N in "${N_VALUES[@]}"; do
  for IT in "${IT_VALUES[@]}"; do

    echo "[INFO] Running double precision N=$N IT=$IT ($RUNS runs)"
    for ((x = 1; x <= RUNS; x++)); do
      if [[ -x "./jacobi_ocl_N${N}_IT${IT}_V2" ]]; then
        ./jacobi_ocl_N${N}_IT${IT}_V2 >>"$RESULTS"
      else
        echo "[WARNING] Missing binary: ./jacobi_ocl_N${N}_IT${IT}_V2" | tee -a missing_binaries.log
      fi
    done

    echo "[INFO] Running float precision N=$N IT=$IT ($RUNS runs)"
    for ((x = 1; x <= RUNS; x++)); do
      if [[ -x "./jacobi_ocl_N${N}_IT${IT}_float_V2" ]]; then
        ./jacobi_ocl_N${N}_IT${IT}_float_V2 >>"$RESULTS"
      else
        echo "[WARNING] Missing binary: ./jacobi_ocl_N${N}_IT${IT}_float_V2" | tee -a missing_binaries.log
      fi
    done

  done
done

echo
echo "[DONE] Results written to ${RESULTS}"
if [[ -f missing_binaries.log ]]; then
  echo "[NOTE] Some binaries were missing — see missing_binaries.log"
fi

echo "[INFO] End time: $(date)"
