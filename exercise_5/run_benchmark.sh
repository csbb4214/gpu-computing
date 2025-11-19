#!/usr/bin/env bash
# Requires: make in PATH, run from repository folder containing the Makefile.

set -euo pipefail

# --- config ---
RESULTS="results.csv"
N_VALUES="1024 1048576 536870912"
RUNS=10

# --- clean build log ---
[ -f build.log ] && rm build.log

# --- CSV header ---
echo "version,precision,N,result,elapsed_ms" >"$RESULTS"

for N in $N_VALUES; do
  #################
  # --- int ---
  #################

  echo "[INFO] Building int N=$N"

  if ! make all N="$N" >build_temp.log 2>&1; then
    echo "[ERROR] Build failed for int N=$N"
    cat build_temp.log >>build.log
    rm -f build_temp.log
    exit 1
  fi

  # Check for warnings
  if grep -qi "warning" build_temp.log; then
    echo "[WARNING] Build warnings for int N=$N"
    cat build_temp.log >>build.log
  fi
  rm -f build_temp.log

  echo "[INFO] Running int binaries..."

  for ((x = 1; x <= RUNS; x++)); do
    ./sequential_reduction_N${N} >>"$RESULTS"
    ./parallel_reduction_N${N} >>"$RESULTS"
    ./multistage_reduction_N${N} >>"$RESULTS"
  done

  ###################
  # --- float ---
  ###################

  echo "[INFO] Building float N=$N"

  if ! make all N="$N" FLOAT=1 >build_temp.log 2>&1; then
    echo "[ERROR] Build failed for float N=$N"
    cat build_temp.log >>build.log
    rm -f build_temp.log
    exit 1
  fi

  # Check for warnings
  if grep -qi "warning" build_temp.log; then
    echo "[WARNING] Build warnings for float N=$N"
    cat build_temp.log >>build.log
  fi
  rm -f build_temp.log

  echo "[INFO] Running float binaries..."

  for ((x = 1; x <= RUNS; x++)); do
    ./sequential_reduction_N${N}_float >>"$RESULTS"
    ./parallel_reduction_N${N}_float >>"$RESULTS"
    ./multistage_reduction_N${N}_float >>"$RESULTS"
  done
done

# --- cleanup ---
sleep 5
make clean

echo "[DONE] Results written to \"$RESULTS\"."
if [ -f build.log ]; then
  echo "[NOTE] Warnings/errors logged to build.log"
fi

exit 0
