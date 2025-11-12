#!/usr/bin/env bash
# Bash script to build all Jacobi variants (double & float)

set -euo pipefail

# --- config ---
N_VALUES=(1024 2048)
IT_VALUES=(10 100 1000)

# --- clean build log ---
rm -f build.log
rm -f build_temp.log

echo "[INFO] Starting build of Jacobi variants..."
echo

for N in "${N_VALUES[@]}"; do
  for IT in "${IT_VALUES[@]}"; do

    # ------------------------
    # --- double precision ---
    # ------------------------
    echo "[INFO] Building double precision: N=$N IT=$IT"

    if ! make all N="$N" IT="$IT" >build_temp.log 2>&1; then
      echo "[ERROR] Build failed for double N=$N IT=$IT"
      cat build_temp.log >>build.log
      rm -f build_temp.log
      exit 1
    fi

    if grep -iq "warning" build_temp.log; then
      echo "[WARNING] Build warnings for double N=$N IT=$IT"
      cat build_temp.log >>build.log
    fi
    rm -f build_temp.log

    # ------------------------
    # --- float precision ---
    # ------------------------
    echo "[INFO] Building float precision: N=$N IT=$IT"

    if ! make all N="$N" IT="$IT" FLOAT=1 >build_temp.log 2>&1; then
      echo "[ERROR] Build failed for float N=$N IT=$IT"
      cat build_temp.log >>build.log
      rm -f build_temp.log
      exit 1
    fi

    if grep -iq "warning" build_temp.log; then
      echo "[WARNING] Build warnings for float N=$N IT=$IT"
      cat build_temp.log >>build.log
    fi
    rm -f build_temp.log

    echo

  done
done

echo "[DONE] All binaries built successfully!"
echo "Check build.log for any warnings or errors."
