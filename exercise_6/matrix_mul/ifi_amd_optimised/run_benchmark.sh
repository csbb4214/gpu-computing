#!/usr/bin/env bash
# Requires: make in PATH, run from repository folder containing the Makefile.

set -euo pipefail

# --- config ---
RESULTS="matrix_mul_results.csv"
N_VALUES="512 1024 2048 2000"
RUNS=10

# --- clean build log ---
[ -f build.log ] && rm build.log

# --- CSV header ---
# impl:       opencl / später evtl. openmp
# precision:  float / double
# N,M,K:      Matrixgrößen (hier alle gleich)
# C00:        C[0,0] zur groben Plausibilitätskontrolle
# elapsed_ms: gemessene Zeit
echo "impl,precision,N,M,K,C00,elapsed_ms" >"$RESULTS"

for N in $N_VALUES; do
  ################################################
  # --------- OpenCL float build & run ----------
  ################################################

  echo "[INFO] Building OpenCL float N=$N"

  if ! make all N="$N" >build_temp.log 2>&1; then
    echo "[ERROR] Build failed for OpenCL float N=$N"
    cat build_temp.log >>build.log
    rm -f build_temp.log
    exit 1
  fi

  # Check for warnings
  if grep -qi "warning" build_temp.log; then
    echo "[WARNING] Build warnings for OpenCL float N=$N"
    cat build_temp.log >>build.log
  fi
  rm -f build_temp.log

  bin_float="matrix_mul_N${N}_float"
  echo "[INFO] Running $bin_float ..."

  for ((x = 1; x <= RUNS; x++)); do
    out=$("./$bin_float")

    # Expect output: C[0,0] = <val>, time = <ms> ms
    C00=$(printf '%s\n' "$out" | sed -n 's/^C\[0,0\] = \([0-9.eE+-]*\), time = .*/\1/p')
    TMS=$(printf '%s\n' "$out" | sed -n 's/^C\[0,0\] = [0-9.eE+-]*, time = \([0-9.eE+-]*\) ms/\1/p')

    if [ -z "$C00" ] || [ -z "$TMS" ]; then
      echo "[ERROR] Failed to parse output of $bin_float:"
      echo "$out"
      exit 1
    fi

    echo "opencl,float,$N,$N,$N,$C00,$TMS" >>"$RESULTS"
  done

  ##################################################
  # --------- OpenCL double build & run -----------
  ##################################################

  echo "[INFO] Building OpenCL double N=$N"

  if ! make all N="$N" USE_DOUBLE=1 >build_temp.log 2>&1; then
    echo "[ERROR] Build failed for OpenCL double N=$N"
    cat build_temp.log >>build.log
    rm -f build_temp.log
    exit 1
  fi

  # Check for warnings
  if grep -qi "warning" build_temp.log; then
    echo "[WARNING] Build warnings for OpenCL double N=$N"
    cat build_temp.log >>build.log
  fi
  rm -f build_temp.log

  bin_double="matrix_mul_N${N}_double"
  echo "[INFO] Running $bin_double ..."

  for ((x = 1; x <= RUNS; x++)); do
    out=$("./$bin_double")

    C00=$(printf '%s\n' "$out" | sed -n 's/^C\[0,0\] = \([0-9.eE+-]*\), time = .*/\1/p')
    TMS=$(printf '%s\n' "$out" | sed -n 's/^C\[0,0\] = [0-9.eE+-]*, time = \([0-9.eE+-]*\) ms/\1/p')

    if [ -z "$C00" ] || [ -z "$TMS" ]; then
      echo "[ERROR] Failed to parse output of $bin_double:"
      echo "$out"
      exit 1
    fi

    echo "opencl,double,$N,$N,$N,$C00,$TMS" >>"$RESULTS"
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
