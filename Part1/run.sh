#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
N=${1:-1024} # Matrix size (N x N), default 1024
P=${2:-4}    # Number of processes/threads, default 4
NUM_RUNS=10  # Number of times to run each implementation

# C++ Compilation settings
CXX=g++
# -O3: Strong optimization [cite: 11]
# -march=native: Use CPU-specific instructions (like AVX) [cite: 11]
# -fopenmp: Enable OpenMP for parallel regions [cite: 5]
CXX_FLAGS="-O3 -march=native -fopenmp -std=c++17"
SRC_FILE="optimized/cpp/gemm_opt.cpp" # [cite: 42]
BIN_FILE="optimized/gemm_opt"     # [cite: 43]
# ---------------------

echo "Starting benchmark:"
echo "  N = $N (Matrix size)"
echo "  P = $P (Processes/Threads)"
echo "  Runs = $NUM_RUNS (per implementation)"
echo "------------------------------------------"

# 1. COMPILE C++ OPTIMIZED CODE
echo "Compiling: $CXX $CXX_FLAGS $SRC_FILE -o $BIN_FILE"
if ! $CXX $CXX_FLAGS $SRC_FILE -o $BIN_FILE; then
    echo "Error: C++ compilation failed."
    exit 1
fi

# Check if compilation was successful
if [ ! -x "$BIN_FILE" ]; then
    echo "Error: Optimized binary '$BIN_FILE' not found or not executable after compile."
    exit 1
fi
echo "Compilation successful."
echo "------------------------------------------"


# 2. Run Baseline (Python)
echo "Running baseline (python3 baseline/gemm_baseline.py)..."
baseline_total=0.0
for i in $(seq 1 $NUM_RUNS); do
    echo -n "  Run $i/$NUM_RUNS... "
    
    # Use /usr/bin/time -p to get POSIX-standard 'real' time output on stderr
    # { ...; } 2>&1 redirects stderr to stdout
    # grep 'real' filters for the line with wall-clock time
    # awk '{print $2}' extracts the second column (the time value)
    run_time=$( { /usr/bin/time -p python3 baseline/gemm_baseline.py $N $P; } 2>&1 | grep 'real' | awk '{print $2}' )
    
    # Add to total using bc for floating point math
    baseline_total=$(echo "$baseline_total + $run_time" | bc)
    echo "$run_time s"
done

# Calculate average baseline time
# Use 'bc -l' or 'scale=' for floating point division
baseline_avg=$(echo "scale=6; $baseline_total / $NUM_RUNS" | bc)
echo "Average baseline time: $baseline_avg s"
echo "------------------------------------------"


# 3. Run Optimized (C++)
echo "Running optimized (./$BIN_FILE)..."
optimized_total=0.0
for i in $(seq 1 $NUM_RUNS); do
    echo -n "  Run $i/$NUM_RUNS... "
    
    # Use the same time-capturing method for the C++ binary
    run_time=$( { /usr/bin/time -p ./$BIN_FILE $N $P; } 2>&1 | grep 'real' | awk '{print $2}' )
    
    optimized_total=$(echo "$optimized_total + $run_time" | bc)
    echo "$run_time s"
done

# Calculate average optimized time
optimized_avg=$(echo "scale=6; $optimized_total / $NUM_RUNS" | bc)
echo "Average optimized time: $optimized_avg s"
echo "------------------------------------------"


# 4. Report Final Speedup
# Check for divide-by-zero, just in case
if (( $(echo "$optimized_avg == 0" | bc -l) )); then
    echo "Error: Optimized time was zero. Cannot calculate speedup."
    exit 1
fi

speedup=$(echo "scale=4; $baseline_avg / $optimized_avg" | bc)

echo "Benchmark Complete"
echo
echo "  Average Baseline Time:  $baseline_avg s"
echo "  Average Optimized Time: $optimized_avg s"
echo
echo "  Speedup (Baseline / Optimized): ${speedup}x"
echo "------------------------------------------"