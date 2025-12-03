#!/bin/bash
#SBATCH --job-name=auto_levels_benchmark_ifi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=10000
#SBATCH --partition=IFIgpu2070
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=auto_levels_run.%j.out
#SBATCH --error=auto_levels_run.%j.err

# Run auto_levels benchmarks and collect results in CSV.
# Erwartung: Binaries wurden vorher mit ./build.sh gebaut!

set -euo pipefail

echo "[INFO] Starting auto_levels benchmark job..."
echo "[INFO] Host: $(hostname)"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] Start time: $(date)"
echo

# --- config ---
RESULTS="auto_levels_results_rtx.csv"
INPUT_IMAGE="earth-huge.png"
OUTPUT_IMAGE_SERIAL="auto_levels_benchmark_serial_out.png"
OUTPUT_IMAGE_OCL="auto_levels_benchmark_ocl_out.png"
RUNS=10

# Validate input exists
if [[ ! -f "$INPUT_IMAGE" ]]; then
  echo "[ERROR] Input image not found: $INPUT_IMAGE"
  exit 1
fi

# Prüfen, ob Binaries existieren
for exe in "./auto_levels" "./auto_levels_cl"; do
  if [[ ! -x "$exe" ]]; then
    echo "[ERROR] Binary $exe not found or not executable."
    echo "        Did you run ./build.sh ?"
    exit 1
  fi
done

# --- clean previous results ---
[ -f "$RESULTS" ] && rm "$RESULTS"

# --- CSV header ---
# impl:       serial, opencl
# elapsed_ms: total time (time_ii: img in host-mem to modified img in host-mem)
# time_ia:    gpu_cpu_time (includes time on cpu between the two kernels)
# time_ib:    gpu_only_time
echo "impl,elapsed_ms,time_ia,time_ib" >"$RESULTS"

echo "[INFO] Running $RUNS iterations for each implementation"

for ((x = 1; x <= RUNS; x++)); do
  echo "[INFO] Run $x"

  # wichtig: srun verwenden, damit SLURM Ressourcen korrekt tracked
  if ! srun ./auto_levels "$INPUT_IMAGE" "$OUTPUT_IMAGE_SERIAL" >>"$RESULTS" 2>&1; then
    echo "[ERROR] Serial version failed"
    exit 1
  fi

  if ! srun ./auto_levels_cl "$INPUT_IMAGE" "$OUTPUT_IMAGE_OCL" >>"$RESULTS" 2>&1; then
    echo "[ERROR] OpenCL version failed"
    exit 1
  fi
done

echo "[DONE] Results written to $RESULTS"
echo "[INFO] End time: $(date)"
