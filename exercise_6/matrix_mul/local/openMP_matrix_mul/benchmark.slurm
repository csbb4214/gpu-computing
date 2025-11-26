#!/bin/bash
#SBATCH --job-name=matrix_mul_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=IFIgpu2070
#SBATCH --gres=gpu:1
#SBATCH --error=matrix_mul_%j.err
#SBATCH --output=matrix_mul_%j.out

# Matrixgrößen
MATRIX_SIZES=(512 1024 2000 2048)

for N in "${MATRIX_SIZES[@]}"; do
    echo "-----------------------------"
    echo "Building matrix_mul with N=$N"

    ./matrix_mul $N
    ./matrix_mul_omp $N
done
# make clean -C "$BASE_DIR"

echo "All tests finished"

