#!/usr/bin/env bash
set -euo pipefail

# This script builds the optimized C++ and then runs a benchmark
# for a SINGLE (N, T) pair against the Python baseline.
#
# It performs (for each version):
# 1. Compilation of the C++ code (with NUMA auto-detection).
# 2. A warmup run.
# 3. A specified number of sample runs (e.g., 3).
# 4. It averages the results and calculates the speedup.
# 5. It appends all metrics to the CSV file.
#
# Usage: ./run.sh <N> <T> <output_csv_file>

# --- Configuration ---
N=${1}
T=${2}
OUT_FILE=${3}

SRC_FILE_OPT="optimized/cpp/gemm_opt.cpp"
BIN_FILE_OPT="optimized/gemm_opt"
BIN_FILE_BASE="baseline/gemm_baseline.py"
NUM_SAMPLES=3 # Number of samples to run and average
# ---------------------

# --- Input Validation ---
if [ -z "$N" ] || [ -z "$T" ] || [ -z "$OUT_FILE" ]; then
    echo "Usage: $0 <N> <T> <output_csv_file>"
    exit 1
fi
if [ ! -f "$SRC_FILE_OPT" ]; then
    echo "Error: Optimized source '$SRC_FILE_OPT' not found." >&2
    exit 1
fi
if [ ! -f "$BIN_FILE_BASE" ]; then
    echo "Error: Baseline script '$BIN_FILE_BASE' not found." >&2
    exit 1
fi

# --- 1. Check for NUMA Development Library & Compile ---
NUMA_CFLAGS=""
NUMA_LIBS=""

# Create a temporary C++ file to test compilation
echo "#include <numa.h>" > .numa_check.cpp
if g++ -E .numa_check.cpp -o /dev/null &> /dev/null; then
    echo "--- Found 'numa.h'. Compiling with NUMA-aware optimizations. ---"
    NUMA_CFLAGS="-DHAS_NUMA" # Define the HAS_NUMA macro for C++
    NUMA_LIBS="-lnuma"       # Link the NUMA library
else
    echo "--- 'numa.h' not found. Building without NUMA-aware optimizations. ---"
fi
rm -f .numa_check.cpp

echo "Building $BIN_FILE_OPT..."
g++ -O3 -march=native -fopenmp -std=c++17 $NUMA_CFLAGS $SRC_FILE_OPT -o $BIN_FILE_OPT $NUMA_LIBS

if [ $? -ne 0 ]; then
    echo "Error: C++ compilation failed." >&2
    exit 1
fi
echo "Build successful."
# ----------------------------------------------------

echo ""
echo "--- Benchmarking N=$N, T=$T (Samples=$NUM_SAMPLES) ---"

# === 2. Run Optimized C++ Benchmark ===
echo "Running Optimized C++ (1 warmup + $NUM_SAMPLES samples)..."
# Warmup
$BIN_FILE_OPT $N $T > /dev/null

total_time_opt=0.0
total_gflops_opt=0.0

for i in $(seq 1 $NUM_SAMPLES); do
    OUTPUT=$($BIN_FILE_OPT $N $T)
    # Use grep and awk to robustly parse the output
    DATA_LINE=$(echo "$OUTPUT" | grep "^N=")
    GFLOPS_LINE=$(echo "$OUTPUT" | grep "^GFLOPS=")

    TIME_VAL=$(echo "$DATA_LINE" | awk -F'[= ]' '{print $6}')
    GFLOPS_VAL=$(echo "$GFLOPS_LINE" | awk -F'[= ]' '{print $2}')
    
    total_time_opt=$(echo "$total_time_opt + $TIME_VAL" | bc)
    total_gflops_opt=$(echo "$total_gflops_opt + $GFLOPS_VAL" | bc)
done

avg_time_opt=$(echo "scale=6; $total_time_opt / $NUM_SAMPLES" | bc)
avg_gflops_opt=$(echo "scale=6; $total_gflops_opt / $NUM_SAMPLES" | bc)
echo "Optimized Avg: Time=${avg_time_opt}s, GFLOPS=${avg_gflops_opt}"

# === 3. Run Baseline Python Benchmark ===
echo "Running Baseline Python (1 warmup + $NUM_SAMPLES samples)..."
# Warmup
( /usr/bin/time -p python3 $BIN_FILE_BASE $N $T ) &> /dev/null

total_time_base=0.0

for i in $(seq 1 $NUM_SAMPLES); do
    # Use /usr/bin/time -p to get 'real' time on stderr
    # { ...; } 2>&1 redirects stderr to stdout
    run_time_base=$( { /usr/bin/time -p python3 $BIN_FILE_BASE $N $T; } 2>&1 | grep 'real' | awk '{print $2}' )
    
    total_time_base=$(echo "$total_time_base + $run_time_base" | bc)
done

avg_time_base=$(echo "scale=6; $total_time_base / $NUM_SAMPLES" | bc)
echo "Baseline Avg: Time=${avg_time_base}s"

# === 4. Calculate Speedup and Log to CSV ===
speedup=0.0
# Check for divide-by-zero, just in case
if (( $(echo "$avg_time_opt > 0" | bc -l) )); then
    speedup=$(echo "scale=4; $avg_time_base / $avg_time_opt" | bc)
fi
echo "Speedup (Base/Opt): ${speedup}x"

# Append all data as a new row in the CSV
echo "$N,$T,$avg_time_opt,$avg_gflops_opt,$avg_time_base,$speedup" >> $OUT_FILE
echo "Result logged to $OUT_FILE"