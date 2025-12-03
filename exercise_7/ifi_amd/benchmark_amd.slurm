#!/bin/bash
#SBATCH --job-name=auto_levels_benchmark_ifi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=10000
#SBATCH --partition=IFIAMD
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=auto_levels_run.%j.out
#SBATCH --error=auto_levels_run.%j.err

# Build and run all auto_levels binaries and collect results in CSV.

set -euo pipefail

echo "[INFO] Starting auto_levels benchmark job..."
echo "[INFO] Host: $(hostname)"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Start time: $(date)"
echo

# --- config ---
RESULTS="auto_levels_results_amd.csv"
INPUT_IMAGE="earth-huge.png"
OUTPUT_IMAGE_SERIAL="auto_levels_benchmark_serial_out.png"
OUTPUT_IMAGE_OCL="auto_levels_benchmark_ocl_out.png"
RUNS=10

# Validate input exists
if [[ ! -f "$INPUT_IMAGE" ]]; then
  echo "[ERROR] Input image not found: $INPUT_IMAGE"
  exit 1
fi

# --- clean previous results ---
[ -f build.log ] && rm build.log
[ -f "$RESULTS" ] && rm "$RESULTS"

# --- CSV header ---
# impl:       serial, opencl
# elapsed_ms: total time (time_ii: img in host-mem to modified img in host-mem)
# time_ia:    gpu_cpu_time (includes time on cpu between the two kernels)
# time_ib:    gpu_only_time
echo "impl,elapsed_ms,time_ia,time_ib" >"$RESULTS"

# --- build binaries ---
echo "[INFO] Building binaries"

if ! make all > build_temp.log 2>&1; then
  echo "[ERROR] Build failed"
  cat build_temp.log >> build.log
  rm -f build_temp.log
  exit 1
fi

if grep -qi "warning" build_temp.log; then
  echo "[WARNING] Build warnings"
  cat build_temp.log >> build.log
fi
rm -f build_temp.log

# --- run benchmark ---
echo "[INFO] Running $RUNS iterations for each implementation"

for ((x = 1; x <= RUNS; x++)); do
  echo "[INFO] Run $x"
  
  if ! ./auto_levels "$INPUT_IMAGE" "$OUTPUT_IMAGE_SERIAL" >> "$RESULTS" 2>&1; then
    echo "[ERROR] Serial version failed"
    exit 1
  fi
  
  if ! ./auto_levels_cl "$INPUT_IMAGE" "$OUTPUT_IMAGE_OCL" >> "$RESULTS" 2>&1; then
    echo "[ERROR] OpenCL version failed"
    exit 1
  fi
done

# --- cleanup ---
make clean

echo "[DONE] Results written to $RESULTS"
if [ -f build.log ]; then
  echo "[NOTE] Warnings/errors logged to build.log"
fi

echo "[INFO] End time: $(date)"
