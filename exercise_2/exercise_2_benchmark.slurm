#!/bin/bash
#SBATCH --job-name=jacobi_benchmark
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=IFIgpu2070
#SBATCH --gres=gpu:1
#SBATCH --error=jacobi_benchmark.%j.err
#SBATCH --output=jacobi_benchmark.%j.out

set -euo pipefail

RESULTS="results.csv"
N_VALUES=(1024 2048)
IT_VALUES=(10 100 1000)

# CSV header (matches Windows script)
echo "mode,precision,N,IT,time_ms,checksum" > "${RESULTS}"

# helper to build and check
build_or_fail() {
  local extra_opts="$1"
  make all ${extra_opts} > build.log 2>&1 || {
    echo "[ERROR] Build failed: make all ${extra_opts} (see build.log)"
    cat build.log
    exit 1
  }
}

for N in "${N_VALUES[@]}"; do
  for IT in "${IT_VALUES[@]}"; do

    echo "[INFO] Building double N=${N} IT=${IT}"
    build_or_fail "N=${N} IT=${IT}"

    echo "[INFO] Running double binaries..."
    ./jacobi_N${N}_IT${IT}          >> "${RESULTS}"
    ./jacobi_omp_N${N}_IT${IT}      >> "${RESULTS}"
    ./jacobi_ocl_N${N}_IT${IT}_V1   >> "${RESULTS}"
    ./jacobi_ocl_N${N}_IT${IT}_V2   >> "${RESULTS}"

    echo "[INFO] Building float N=${N} IT=${IT}"
    build_or_fail "N=${N} IT=${IT} FLOAT=1"

    echo "[INFO] Running float binaries..."
    ./jacobi_N${N}_IT${IT}_float        >> "${RESULTS}"
    ./jacobi_omp_N${N}_IT${IT}_float    >> "${RESULTS}"
    ./jacobi_ocl_N${N}_IT${IT}_float_V1 >> "${RESULTS}"
    ./jacobi_ocl_N${N}_IT${IT}_float_V2 >> "${RESULTS}"

  done
done

make clean > /dev/null 2>&1 || true

echo "[DONE] Results written to ${RESULTS}"