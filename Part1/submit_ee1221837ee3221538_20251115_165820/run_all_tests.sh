#!/usr/bin/env bash
set -euo pipefail

N_VALUES=(512 1024 1536 2048)
T_VALUES=(1 2 4 8 16)

OUT_FILE="gemm_results.csv"
MODE="full"

echo "Initializing $OUT_FILE..."

# NEW HEADER
echo "N,T,Time_Opt,GFLOPS_Opt,Time_Base,Speedup" > $OUT_FILE

echo "Starting tests..."

for N in "${N_VALUES[@]}"; do
    for T in "${T_VALUES[@]}"; do
        ./run_tests.sh $MODE $N $T
    done
done

echo "All tests complete."
