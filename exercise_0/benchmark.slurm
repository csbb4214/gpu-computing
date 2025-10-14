#!/bin/bash

#SBATCH --job-name=array_sum_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=IFIgpu2070
#SBATCH --gres=gpu:1
#SBATCH --error=array_sum_%j.err

# Array sizes
ARRAY_SIZES=(100000 1000000 10000000 100000000)

echo "Running array_sum tests on $(hostname)"

for SIZE in "${ARRAY_SIZES[@]}"
do
    echo "-----------------------------"
    echo "Array size: $SIZE"
    # Simple time measurement
    time ./array_sum $SIZE
done

echo "All tests finished"

