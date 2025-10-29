#!/bin/bash
#SBATCH --job-name=jacobi_benchmark
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=10000
#SBATCH --partition=IFIgpu2070
#SBATCH --gres=gpu:1
#SBATCH --error=jacobi_benchmark.%j.err
#SBATCH --output=jacobi_benchmark.%j.out

set -euo pipefail

RESULTS="results.csv"
N_VALUES=(1024 2048)
IT_VALUES=(10 100 1000)

echo "mode,precision,N,IT,time_ms,checksum" > "${RESULTS}"

for N in "${N_VALUES[@]}"; do
  for IT in "${IT_VALUES[@]}"; do

    echo "[INFO] Running double N=${N} IT=${IT}"
    ./jacobi_N${N}_IT${IT}          >> "${RESULTS}"
    ./jacobi_omp_N${N}_IT${IT}      >> "${RESULTS}"
    ./jacobi_ocl_N${N}_IT${IT}_V1   >> "${RESULTS}"
    ./jacobi_ocl_N${N}_IT${IT}_V2   >> "${RESULTS}"

    echo "[INFO] Running float N=${N} IT=${IT}"
    ./jacobi_N${N}_IT${IT}_float        >> "${RESULTS}"
    ./jacobi_omp_N${N}_IT${IT}_float    >> "${RESULTS}"
    ./jacobi_ocl_N${N}_IT${IT}_float_V1 >> "${RESULTS}"
    ./jacobi_ocl_N${N}_IT${IT}_float_V2 >> "${RESULTS}"

  done
done

echo "[DONE] Results written to ${RESULTS}"

