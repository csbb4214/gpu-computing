#!/bin/bash
#SBATCH --job-name=jacobi_benchmark_ifi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=10000
#SBATCH --partition=IFIAMD
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=jacobi_v3.%j.out
#SBATCH --error=jacobi_v3.%j.err

# Run pre-compiled Jacobi binaries and collect results in CSV.

set -euo pipefail

echo "[INFO] Starting Jacobi benchmark..."
echo "[INFO] Host: $(hostname)"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Start time: $(date)"
echo

# --- config ---
RESULTS="results.csv"
BUILD_DIR="binaries"
N_VALUES=(2048 4096)
IT_VALUES=(10 100 1000)
DIM1_VALUES=(8 4 2 1)
DIM2_VALUES=(32 64 128 256)
RUNS=5

# --- verify binaries exist ---
if [[ ! -d "$BUILD_DIR" ]]; then
  echo "[ERROR] Binary directory not found: $BUILD_DIR"
  echo "        Please run build_all.sh first!"
  exit 1
fi

BINARY_COUNT=$(ls "$BUILD_DIR"/jacobi_ocl_* 2>/dev/null | wc -l)
if [[ $BINARY_COUNT -eq 0 ]]; then
  echo "[ERROR] No binaries found in $BUILD_DIR/"
  echo "        Please run build_all.sh first!"
  exit 1
fi

echo "[INFO] Found $BINARY_COUNT binaries in $BUILD_DIR/"
echo

# --- cleanup old results ---
rm -f "$RESULTS" run_errors.log

# --- CSV header ---
echo "version,precision,N,IT,LOCAL_WORKGROUP_DIM_1,LOCAL_WORKGROUP_DIM_2,elapsed_ms" >"$RESULTS"

# --- run benchmarks ---
TOTAL=0
SUCCESS=0
FAILED=0

for N in "${N_VALUES[@]}"; do
  for IT in "${IT_VALUES[@]}"; do
    echo "[INFO] N=$N IT=$IT"

    for ((i = 0; i<${#DIM1_VALUES[@]}; i++)); do
      D1=${DIM1_VALUES[$i]}
      D2=${DIM2_VALUES[$i]}

      # --- double ---
      BIN="$BUILD_DIR/jacobi_ocl_N${N}_IT${IT}_DIM1-${D1}_DIM2-${D2}_V3"
      
      if [[ -x "$BIN" ]]; then
        echo "  [RUN] double (D1=$D1, D2=$D2)"
        for ((x = 1; x <= RUNS; x++)); do
          TOTAL=$((TOTAL + 1))
          if "$BIN" >> "$RESULTS" 2>> run_errors.log; then
            SUCCESS=$((SUCCESS + 1))
          else
            FAILED=$((FAILED + 1))
            echo "[ERROR] Run $x/$RUNS failed for $BIN" | tee -a run_errors.log
            echo "FAILED,double,$N,$IT,$D1,$D2,0" >> "$RESULTS"
          fi
        done
      else
        echo "  [WARNING] Missing binary: $BIN"
      fi

      # --- float ---
      BIN="$BUILD_DIR/jacobi_ocl_N${N}_IT${IT}_DIM1-${D1}_DIM2-${D2}_float_V3"
      
      if [[ -x "$BIN" ]]; then
        echo "  [RUN] float (D1=$D1, D2=$D2)"
        for ((x = 1; x <= RUNS; x++)); do
          TOTAL=$((TOTAL + 1))
          if "$BIN" >> "$RESULTS" 2>> run_errors.log; then
            SUCCESS=$((SUCCESS + 1))
          else
            FAILED=$((FAILED + 1))
            echo "[ERROR] Run $x/$RUNS failed for $BIN" | tee -a run_errors.log
            echo "FAILED,float,$N,$IT,$D1,$D2,0" >> "$RESULTS"
          fi
        done
      else
        echo "  [WARNING] Missing binary: $BIN"
      fi

      echo
    done
  done
done

echo "[SUMMARY] Benchmark complete at $(date)"
echo "  Total runs: $TOTAL"
echo "  Successful: $SUCCESS"
echo "  Failed: $FAILED"
echo "  Results: $RESULTS"

if [[ $FAILED -gt 0 ]]; then
  echo "  Errors logged to: run_errors.log"
  echo
  echo "[ERROR SUMMARY]"
  tail -20 run_errors.log
fi