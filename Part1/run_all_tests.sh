#!/usr/bin/env bash
set -euo pipefail

# --- Test Parameters (Customize These) ---
# Matrix Sizes (N x N)
N_VALUES=(512 1024 1536 2048)
# Thread Counts (T)
T_VALUES=(1 2 4 8 16)
# ---

# Output CSV file
OUT_FILE="gemm_results.csv"

# C++ Compilation settings
CXX=g++
CXX_FLAGS="-O3 -march=native -fopenmp -std=c++17"
SRC_FILE="optimized/cpp/gemm_opt.cpp" # [cite: 42]
BIN_FILE="optimized/gemm_opt"     # [cite: 43]

# 1. COMPILE C++ OPTIMIZED CODE (ONCE)
echo "Compiling: $CXX $CXX_FLAGS $SRC_FILE -o $BIN_FILE"
if ! $CXX $CXX_FLAGS $SRC_FILE -o $BIN_FILE; then
    echo "Error: C++ compilation failed."
    exit 1
fi
echo "Compilation successful."

# 2. CREATE CSV HEADER
echo "Initializing $OUT_FILE..."
echo "N,T,Time,GFLOPS" > $OUT_FILE

# 3. RUN ALL TESTS
echo "Starting benchmark run..."
# Loop over N values
for N in "${N_VALUES[@]}"; do
    # Loop over T values
    for T in "${T_VALUES[@]}"; do
        # Run the single test script,
        # which will append its result to $OUT_FILE
        ./run.sh $N $T $OUT_FILE
    done
done

echo "------------------------------------------"
echo "All tests complete."
echo "Results saved in $OUT_FILE"
echo "You can now run 'python3 plot_results.py' to visualize."