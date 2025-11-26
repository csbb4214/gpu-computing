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

# --- config ---
RESULTS="matrix_mul_results.csv"
N_VALUES=(512 1024 2048 2000)
RUNS=10

# --- cleanup old result / log files ---
rm -f "$RESULTS" 2>/dev/null || true
rm -f missing_binaries.log 2>/dev/null || true

# --- CSV header (wie im lokalen Benchmark-Skript) ---
# impl,precision,N,M,K,C00,elapsed_ms
echo "impl,precision,N,M,K,C00,elapsed_ms" >"$RESULTS"

# --- run loop ---
for N in "${N_VALUES[@]}"; do
  echo "[INFO] N=$N ($RUNS runs each binary)"

  ###################
  # --- float ---
  ###################
  bin_float="./matrix_mul_N${N}_float"
  echo "[INFO] Running OpenCL float binary for N=$N: $bin_float"

  if [[ ! -x "$bin_float" ]]; then
    echo "[WARNING] Missing binary: $bin_float" | tee -a missing_binaries.log
  else
    for ((x = 1; x <= RUNS; x++)); do
      out=$("$bin_float")

      # Expect output: C[0,0] = <val>, time = <ms> ms
      C00=$(printf '%s\n' "$out" | sed -n 's/^C\[0,0\] = \([0-9.eE+-]*\), time = .*/\1/p')
      TMS=$(printf '%s\n' "$out" | sed -n 's/^C\[0,0\] = [0-9.eE+-]*, time = \([0-9.eE+-]*\) ms/\1/p')

      if [[ -z "$C00" || -z "$TMS" ]]; then
        echo "[ERROR] Failed to parse output of $bin_float:"
        echo "$out"
        exit 1
      fi

      echo "opencl,float,$N,$N,$N,$C00,$TMS" >>"$RESULTS"
    done
  fi

  #####################
  # --- double ---
  #####################
  bin_double="./matrix_mul_N${N}_double"
  echo "[INFO] Running OpenCL double binary for N=$N: $bin_double"

  if [[ ! -x "$bin_double" ]]; then
    echo "[WARNING] Missing binary: $bin_double" | tee -a missing_binaries.log
  else
    for ((x = 1; x <= RUNS; x++)); do
      out=$("$bin_double")

      C00=$(printf '%s\n' "$out" | sed -n 's/^C\[0,0\] = \([0-9.eE+-]*\), time = .*/\1/p')
      TMS=$(printf '%s\n' "$out" | sed -n 's/^C\[0,0\] = [0-9.eE+-]*, time = \([0-9.eE+-]*\) ms/\1/p')

      if [[ -z "$C00" || -z "$TMS" ]]; then
        echo "[ERROR] Failed to parse output of $bin_double:"
        echo "$out"
        exit 1
      fi

      echo "opencl,double,$N,$N,$N,$C00,$TMS" >>"$RESULTS"
    done
  fi
done

echo
echo "[DONE] Results written to ${RESULTS}"
if [[ -f missing_binaries.log ]]; then
  echo "[NOTE] Some binaries were missing — see missing_binaries.log"
fi

echo "[INFO] End time: $(date)"
