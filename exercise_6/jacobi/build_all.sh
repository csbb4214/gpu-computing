#!/bin/bash
# build_all.sh - Compile all Jacobi binaries

set -euo pipefail

echo "[INFO] Starting build at $(date)"
echo "[INFO] Host: $(hostname)"

# --- config ---
N_VALUES=(2048 4096)
IT_VALUES=(10 100 1000)
DIM1_VALUES=(8 4 2 1)
DIM2_VALUES=(32 64 128 256)

BUILD_DIR="binaries"
mkdir -p "$BUILD_DIR"

# --- cleanup old builds ---
rm -f build.log
make clean > /dev/null 2>&1 || true

TOTAL=0
SUCCESS=0
FAILED=0

# --- build all configurations ---
for N in "${N_VALUES[@]}"; do
  for IT in "${IT_VALUES[@]}"; do
    for ((i = 0; i<${#DIM1_VALUES[@]}; i++)); do
      D1=${DIM1_VALUES[$i]}
      D2=${DIM2_VALUES[$i]}

      # --- double precision ---
      TOTAL=$((TOTAL + 1))
      BIN="jacobi_ocl_N${N}_IT${IT}_DIM1-${D1}_DIM2-${D2}_V3"
      echo -n "[$TOTAL] Building $BIN (double)... "

      if make all N="$N" IT="$IT" LOCAL_WORKGROUP_DIM_1="$D1" LOCAL_WORKGROUP_DIM_2="$D2" > build_temp.log 2>&1; then
        mv "$BIN" "$BUILD_DIR/"
        echo "OK"
        SUCCESS=$((SUCCESS + 1))
      else
        echo "FAILED"
        echo "=== Failed: $BIN ===" >> build.log
        cat build_temp.log >> build.log
        FAILED=$((FAILED + 1))
      fi

      if grep -iq "warning" build_temp.log 2>/dev/null; then
        echo "  (warnings logged)"
        cat build_temp.log >> build.log
      fi

      rm -f build_temp.log
      make clean > /dev/null 2>&1

      # --- float precision ---
      TOTAL=$((TOTAL + 1))
      BIN="jacobi_ocl_N${N}_IT${IT}_DIM1-${D1}_DIM2-${D2}_float_V3"
      echo -n "[$TOTAL] Building $BIN (float)... "

      if make all N="$N" IT="$IT" FLOAT=1 LOCAL_WORKGROUP_DIM_1="$D1" LOCAL_WORKGROUP_DIM_2="$D2" > build_temp.log 2>&1; then
        mv "$BIN" "$BUILD_DIR/"
        echo "OK"
        SUCCESS=$((SUCCESS + 1))
      else
        echo "FAILED"
        echo "=== Failed: $BIN ===" >> build.log
        cat build_temp.log >> build.log
        FAILED=$((FAILED + 1))
      fi

      if grep -iq "warning" build_temp.log 2>/dev/null; then
        echo "  (warnings logged)"
        cat build_temp.log >> build.log
      fi

      rm -f build_temp.log
      make clean > /dev/null 2>&1
    done
  done
done

echo
echo "[SUMMARY] Build complete at $(date)"
echo "  Total: $TOTAL"
echo "  Success: $SUCCESS"
echo "  Failed: $FAILED"
echo "  Binaries in: $BUILD_DIR/"

if [[ $FAILED -gt 0 ]]; then
  echo "  Errors logged to: build.log"
  exit 1
fi

# Create manifest for verification
ls -lh "$BUILD_DIR/" > "$BUILD_DIR/manifest.txt"
echo "[INFO] Created manifest: $BUILD_DIR/manifest.txt"