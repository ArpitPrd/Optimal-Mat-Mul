#!/usr/bin/env bash
set -euo pipefail

N_VALUES=(512 1024 1536 2048)
T_VALUES=(1 2 4 8 16)

OUT_FILE="gemm_results.csv"

echo "Initializing $OUT_FILE..."

# NEW HEADER
echo "N,T,time_opt,gflops_opt,time_base,speedup,cycles,instr,ipc,l1_load,l1_miss,l1_miss_rate,llc_load,llc_miss,llc_miss_rate,branches,branch_miss,branch_miss_rate" > $OUT_FILE

echo "Starting tests..."

for N in "${N_VALUES[@]}"; do
    for T in "${T_VALUES[@]}"; do
        ./run.sh $N $T $OUT_FILE
    done
done

echo "All tests complete."
