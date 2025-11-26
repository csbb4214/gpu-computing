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

# Run all Jacobi binaries and collect results in CSV.

set -euo pipefail

echo "[INFO] Starting Jacobi run job..."
echo "[INFO] Host: $(hostname)"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Start time: $(date)"
echo

# --- config ---
RESULTS="results.csv"
N_VALUES=(2048 4096)
IT_VALUES=(10 100 1000)
DIM1_VALUES=(8 4 2 1 2)
DIM2_VALUES=(32 64 128 256 256)
RUNS=5

# --- cleanup ---
rm -f build.log build_temp.log
rm -f "$RESULTS" 2>/dev/null || true

# --- CSV header ---
echo "version,precision,N,IT,LOCAL_WORKGROUP_DIM_1,LOCAL_WORKGROUP_DIM_2,elapsed_ms" >"$RESULTS"

# --- build and run binaries ---
for N in "${N_VALUES[@]}"; do
  for IT in "${IT_VALUES[@]}"; do

    echo "[INFO] N=$N IT=$IT"

    for ((i = 0; i<${#DIM1_VALUES[@]}; i++)); do
      D1=${DIM1_VALUES[$i]}
      D2=${DIM2_VALUES[$i]}

      # --- double ---
      echo "[BUILD] double"

      if ! make all N="$N" IT="$IT" LOCAL_WORKGROUP_DIM_1="$D1" LOCAL_WORKGROUP_DIM_2="$D2" > build_temp.log 2>&1; then
        echo "[ERROR] Build failed for double N=$N IT=$IT D1=$D1 D2=$D2"
        cat build_temp.log >> build.log
        exit 1
      fi

      if grep -iq "warning" build_temp.log; then
        echo "[WARNING] Build warnings for double N=$N IT=$IT"
        cat build_temp.log >>build.log
      fi

      BIN="./jacobi_ocl_N${N}_IT${IT}_DIM1-${D1}_DIM2-${D2}_V3"

      if [[ -x "$BIN" ]]; then
        echo "[RUN] double"
        for ((x = 1; x <= RUNS; x++)); do
          $BIN >> "$RESULTS"
        done
      else
        echo "[WARNING] Missing binary: $BIN"
      fi

      rm -f build_temp.log
      make clean > /dev/null

      # --- float ---
      echo "[BUILD] float"

      if ! make all N="$N" IT="$IT" FLOAT=1 LOCAL_WORKGROUP_DIM_1="$D1" LOCAL_WORKGROUP_DIM_2="$D2" > build_temp.log 2>&1; then
        echo "[ERROR] Build failed for float N=$N IT=$IT FLOAT=1 D1=$D1 D2=$D2"
        cat build_temp.log >> build.log
        exit 1
      fi

      if grep -iq "warning" build_temp.log; then
        echo "[WARNING] Build warnings for float N=$N IT=$IT"
        cat build_temp.log >>build.log
      fi

      BIN="./jacobi_ocl_N${N}_IT${IT}_DIM1-${D1}_DIM2-${D2}_float_V3"

      if [[ -x "$BIN" ]]; then
        echo "[RUN] float"
        for ((x = 1; x <= RUNS; x++)); do
          $BIN >> "$RESULTS"
        done
      else
        echo "[WARNING] Missing binary: $BIN"
      fi

      rm -f build_temp.log
      make clean > /dev/null

      echo
    done
  done
done

echo "[DONE] Results written to ${RESULTS}"
echo "[INFO] End time: $(date)"
