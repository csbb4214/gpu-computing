#!/usr/bin/env bash
# Bash script to build/run jacobi variants and collect CSV results.
# Requires: make, OpenCL runtime, run from the repo root containing the Makefile.

set -euo pipefail

# --- config ---
RESULTS="results.csv"
N_VALUES=(1024 2048)
IT_VALUES=(10 100 1000)
RUNS=10

# --- clean build log ---
rm -f build.log

# --- clean detail files ---
rm -f kernel_times_*.csv 2>/dev/null || true

# --- CSV header ---
echo "mode,precision,N,IT,total_write,total_kernel,total_read,write_f,write_tmp,write_u" >"$RESULTS"

for N in "${N_VALUES[@]}"; do
  for IT in "${IT_VALUES[@]}"; do

    # ------------------------
    # --- double precision ---
    # ------------------------
    echo "[INFO] Building double N=$N IT=$IT"

    if ! make all N="$N" IT="$IT" >build_temp.log 2>&1; then
      echo "[ERROR] Build failed for double N=$N IT=$IT"
      cat build_temp.log >>build.log
      rm -f build_temp.log
      exit 1
    fi

    # Check for warnings
    if grep -iq "warning" build_temp.log; then
      echo "[WARNING] Build warnings for double N=$N IT=$IT"
      cat build_temp.log >>build.log
    fi
    rm -f build_temp.log

    echo "[INFO] Running double binaries ($RUNS runs)..."
    for ((x = 1; x <= RUNS; x++)); do
      ./jacobi_ocl_N${N}_IT${IT}_V2 >>"$RESULTS"
    done

    # -----------------------
    # --- float precision ---
    # -----------------------
    echo "[INFO] Building float N=$N IT=$IT"

    if ! make all N="$N" IT="$IT" FLOAT=1 >build_temp.log 2>&1; then
      echo "[ERROR] Build failed for float N=$N IT=$IT"
      cat build_temp.log >>build.log
      rm -f build_temp.log
      exit 1
    fi

    # Check for warnings
    if grep -iq "warning" build_temp.log; then
      echo "[WARNING] Build warnings for float N=$N IT=$IT"
      cat build_temp.log >>build.log
    fi
    rm -f build_temp.log

    echo "[INFO] Running float binaries ($RUNS runs)..."
    for ((x = 1; x <= RUNS; x++)); do
      ./jacobi_ocl_N${N}_IT${IT}_float_V2 >>"$RESULTS"
    done

  done
done

# --- cleanup ---
sleep 5
make clean

echo "[DONE] Results written to ${RESULTS}."
if [[ -f build.log ]]; then
  echo "[NOTE] Warnings/errors logged to build.log"
fi
