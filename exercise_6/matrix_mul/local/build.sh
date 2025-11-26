#!/usr/bin/env bash
# Build all matrix_mul binaries (float + double) for all N values.
# Run on login node or locally in the repo with the Makefile.

set -euo pipefail

# --- config ---
N_VALUES="512 1024 2048 2000"

# --- clean build log ---
[ -f build.log ] && rm build.log

for N in $N_VALUES; do
  ###################
  # --- float ---
  ###################

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

  #####################
  # --- double ---
  #####################

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

done

echo
echo "[DONE] All matrix_mul binaries built for N_VALUES=($N_VALUES)"
if [ -f build.log ]; then
  echo "[NOTE] Warnings/errors logged to build.log"
fi

exit 0
