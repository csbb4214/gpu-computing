#!/bin/bash
set -euo pipefail

echo "[INFO] Building matrix multiplication benchmarks"

# ---------------- CONFIG ----------------
MATRIX_SIZES=(512 1024 2000 2048)
# ----------------------------------------

echo "[INFO] Cleaning old builds"
make clean

for N in "${MATRIX_SIZES[@]}"; do
  echo "[INFO] Building N=$N (float)"

  make \
    N=$N \
    FLAGS="" \
    SUFFIX="_N${N}_float"

  echo "[INFO] Building N=$N (double)"

  make \
    N=$N \
    FLAGS="-DUSE_DOUBLE" \
    SUFFIX="_N${N}_double"
done

echo
echo "[DONE] All benchmarks built successfully"
