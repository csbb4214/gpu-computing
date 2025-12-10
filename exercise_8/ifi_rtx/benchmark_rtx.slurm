#!/bin/bash
#SBATCH --job-name=scan_benchmark_ifi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=10000
#SBATCH --partition=IFIgpu2070
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=scan_run.%j.out
#SBATCH --error=scan_run.%j.err

# Run scan benchmarks and collect results in CSV.
# Erwartung: Binaries wurden vorher mit ./build.sh gebaut!

set -euo pipefail

echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"

RUNS="${RUNS:-10}"
SKIP_HUGE="${SKIP_HUGE:-0}"

RAW="scan_benchmark_int_rtx.csv"
KERNEL_FILE="./scan.cl"

# Assignment sizes
N1=1024
N2=1048576
N3=536870912

if [[ "$SKIP_HUGE" == "1" ]]; then
  NS=("$N1" "$N2")
else
  NS=("$N1" "$N2" "$N3")
fi

# Sanity check kernel file (host code loads this)
if [[ ! -f "$KERNEL_FILE" ]]; then
  echo "[ERROR] Kernel file not found: $KERNEL_FILE"
  exit 1
fi

# Prepare CSV
rm -f "$RAW"
HOST="$(hostname 2>/dev/null || echo unknown)"
echo "run,impl,elapsed_ms,N,type,host" >"$RAW"

echo "[INFO] RUNS=$RUNS"
echo "[INFO] SIZES=${NS[*]}"
echo "[INFO] Writing CSV: $RAW"
echo ""

# Run loops (int only)
for n in "${NS[@]}"; do
  BIN="./scan_N${n}"

  if [[ ! -x "$BIN" ]]; then
    echo "[ERROR] Expected binary not found or not executable: $BIN"
    echo "        Please run your build.sh beforehand."
    exit 1
  fi

  echo "[INFO] Running $RUNS iterations for N=$n (int)"

  for ((i = 1; i <= RUNS; i++)); do
    OUT="$($BIN 2>&1)" || {
      echo "[ERROR] Program failed on run $i (N=$n)"
      echo "-------- output --------"
      echo "$OUT"
      echo "------------------------"
      exit 1
    }

    SEQ_MS="$(echo "$OUT" | sed -n 's/.*Sequential Time: \([0-9.]\+\) ms.*/\1/p' | head -n 1)"
    CL_MS="$(echo "$OUT" | sed -n 's/.*OpenCL Time: \([0-9.]\+\) ms.*/\1/p' | head -n 1)"

    if [[ -z "$SEQ_MS" || -z "$CL_MS" ]]; then
      echo "[ERROR] Could not parse timing output on run $i (N=$n)"
      echo "-------- output --------"
      echo "$OUT"
      echo "------------------------"
      exit 1
    fi

    echo "$i,sequential,$SEQ_MS,$n,int,$HOST" >>"$RAW"
    echo "$i,opencl,$CL_MS,$n,int,$HOST" >>"$RAW"
  done

  echo ""
done

echo "[DONE] RTX benchmark finished"
echo "[DONE] CSV: $RAW"
