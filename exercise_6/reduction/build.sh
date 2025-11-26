#!/usr/bin/env bash
# Build all reduction binaries (int + float) for all N values.
# Run on login node oder lokal im Repo mit dem Makefile.

set -euo pipefail

# --- config ---
N_VALUES="1024 1048576 536870912"

# --- clean build log ---
[ -f build.log ] && rm build.log

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

done

echo
echo "[DONE] All binaries built for N_VALUES=($N_VALUES)"
if [ -f build.log ]; then
  echo "[NOTE] Warnings/errors logged to build.log"
fi

exit 0
