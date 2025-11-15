#!/usr/bin/env bash
set -euo pipefail

# This script builds and runs the benchmark in different modes.
#
# Usage: ./run.sh <mode> <N> <T>
#
# Modes:
#   optimized: Compiles and runs the optimized C++ code once.
#   baseline:  Runs the baseline Python code once (with timing).
#   full:      Runs the full benchmark (compile, warmup, samples)
#              and logs results to the CSV file.
# -----------------------------------------------------------------

# --- Configuration ---
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <mode> <N> <T>"
    echo "Modes: optimized, baseline, full"
    exit 1
fi

MODE=${1}
N=${2}
T=${3}

# 'full' mode will log to this file
OUT_FILE="gemm_results.csv"
SRC_FILE_OPT="optimized/cpp/gemm_opt.cpp"
BIN_FILE_OPT="optimized/gemm_opt"
BIN_FILE_BASE="baseline/gemm_baseline.py"
NUM_SAMPLES=3 # Number of samples for 'full' mode
# ---------------------

# --- 1. Compile Function (used by 'optimized' and 'full' modes) ---
compile_optimized() {
    # Check for source file
    if [ ! -f "$SRC_FILE_OPT" ]; then
        echo "Error: Optimized source '$SRC_FILE_OPT' not found." >&2
        exit 1
    fi

    local NUMA_CFLAGS=""
    local NUMA_LIBS=""

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
}

# --- 2. Single Run Functions (for 'optimized' and 'baseline' modes) ---
run_optimized_once() {
    echo "--- Running Optimized C++ (Single Run) N=$N, T=$T ---"
    $BIN_FILE_OPT $N $T
    echo "-----------------------------------------------------"
}

run_baseline_once() {
    if [ ! -f "$BIN_FILE_BASE" ]; then
        echo "Error: Baseline script '$BIN_FILE_BASE' not found." >&2
        exit 1
    fi
    echo "--- Running Baseline Python (Single Run) N=$N, T=$T ---"
    # Use /usr/bin/time -p to show timing, output goes to stderr
    /usr/bin/time -p python3 $BIN_FILE_BASE $N $T
    echo "-----------------------------------------------------"
}

# --- 3. Full Benchmark Function (for 'full' mode) ---
run_full_benchmark() {
    if [ ! -f "$BIN_FILE_BASE" ]; then
        echo "Error: Baseline script '$BIN_FILE_BASE' not found." >&2
        exit 1
    fi

    echo ""
    echo "--- Benchmarking N=$N, T=$T (Samples=$NUM_SAMPLES) ---"

    # === 3a. Run Optimized C++ Benchmark ===
    echo "Running Optimized C++ (1 warmup + $NUM_SAMPLES samples)..."
    # Warmup
    $BIN_FILE_OPT $N $T > /dev/null

    local total_time_opt=0.0
    local total_gflops_opt=0.0

    for i in $(seq 1 $NUM_SAMPLES); do
        local OUTPUT=$($BIN_FILE_OPT $N $T)
        # Use grep and awk to robustly parse the output
        local DATA_LINE=$(echo "$OUTPUT" | grep "^N=")
        local GFLOPS_LINE=$(echo "$OUTPUT" | grep "^GFLOPS=")

        local TIME_VAL=$(echo "$DATA_LINE" | awk -F'[= ]' '{print $6}')
        local GFLOPS_VAL=$(echo "$GFLOPS_LINE" | awk -F'[= ]' '{print $2}')
        
        total_time_opt=$(echo "$total_time_opt + $TIME_VAL" | bc)
        total_gflops_opt=$(echo "$total_gflops_opt + $GFLOPS_VAL" | bc)
    done

    local avg_time_opt=$(echo "scale=6; $total_time_opt / $NUM_SAMPLES" | bc)
    local avg_gflops_opt=$(echo "scale=6; $total_gflops_opt / $NUM_SAMPLES" | bc)
    echo "Optimized Avg: Time=${avg_time_opt}s, GFLOPS=${avg_gflops_opt}"

    # === 3b. Run Baseline Python Benchmark ===
    echo "Running Baseline Python (1 warmup + $NUM_SAMPLES samples)..."
    # Warmup
    ( /usr/bin/time -p python3 $BIN_FILE_BASE $N $T ) &> /dev/null

    local total_time_base=0.0

    for i in $(seq 1 $NUM_SAMPLES); do
        # Use /usr/bin/time -p to get 'real' time on stderr
        # { ...; } 2>&1 redirects stderr to stdout
        local run_time_base=$( { /usr/bin/time -p python3 $BIN_FILE_BASE $N $T; } 2>&1 | grep 'real' | awk '{print $2}' )
        
        total_time_base=$(echo "$total_time_base + $run_time_base" | bc)
    done

    local avg_time_base=$(echo "scale=6; $total_time_base / $NUM_SAMPLES" | bc)
    echo "Baseline Avg: Time=${avg_time_base}s"

    # === 3c. Calculate Speedup and Log to CSV ===
    local speedup=0.0
    # Check for divide-by-zero, just in case
    if (( $(echo "$avg_time_opt > 0" | bc -l) )); then
        speedup=$(echo "scale=4; $avg_time_base / $avg_time_opt" | bc)
    fi
    echo "Speedup (Base/Opt): ${speedup}x"

    # Append all data as a new row in the CSV
    echo "$N,$T,$avg_time_opt,$avg_gflops_opt,$avg_time_base,$speedup" >> $OUT_FILE
    echo "Result logged to $OUT_FILE"
}


# --- 4. Main Execution Logic ---
case "$MODE" in
    optimized)
        # compile_optimized
        run_optimized_once
        ;;
    
    baseline)
        run_baseline_once
        ;;
        
    full)
        # compile_optimized
        run_full_benchmark
        ;;
        
    *)
        echo "Error: Invalid mode '$MODE'."
        echo "Usage: $0 <mode> <N> <T>"
        echo "Modes: optimized, baseline, full"
        exit 1
        ;;
esac