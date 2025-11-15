#!/usr/bin/env bash
set -e

# This script runs targeted 'perf stat' commands to gather profiling
# evidence for the lab report.
#
# It specifically tests the hypotheses for the "weird" N=512 curve
# and the "memory-bound" N=2048 curve.

BIN_FILE="optimized/gemm_opt"

if [ ! -x "$BIN_FILE" ]; then
    echo "Error: Optimized binary '$BIN_FILE' not found." >&2
    echo "Please compile it first (e.g., using run.sh)" >&2
    exit 1
fi

echo "========================================================================"
echo "Hypothesis 1: Analyzing N=512 (AVX-512 Frequency Scaling)"
echo "We are checking if the 'GHz' (clock speed) dips at 8 threads."
echo "========================================================================"

echo "--- Profiling N=512, T=4 (The 'sweet spot') ---"
# -r 3 runs the command 3 times and averages the results for stability
perf stat -r 3 $BIN_FILE 512 4
echo "------------------------------------------------"

echo "--- Profiling N=512, T=8 (The 'dip') ---"
perf stat -r 3 $BIN_FILE 512 8
echo "------------------------------------------------"

echo "--- Profiling N=512, T=16 (The 'recovery') ---"
perf stat -r 3 $BIN_FILE 512 16
echo "------------------------------------------------"


echo ""
echo "========================================================================"
echo "Hypothesis 2: Analyzing N=2048 (Memory-Bound)"
echo "We are checking for low IPC and high cache misses."
echo "========================================================================"

# -e flag lets us specify exact events to monitor.
# LLC-load-misses = Last-Level Cache (L3) misses, the most important one.
# L1-dcache-load-misses = L1 data cache misses.
EVENTS="cycles,instructions,cache-misses,L1-dcache-load-misses,LLC-load-misses"

echo "--- Profiling N=2048, T=1 (Baseline) ---"
perf stat -e $EVENTS -r 3 $BIN_FILE 2048 1
echo "------------------------------------------------"

echo "--- Profiling N=2048, T=16 (Scaled) ---"
perf stat -e $EVENTS -r 3 $BIN_FILE 2048 16
echo "------------------------------------------------"

echo "Profiling complete. See analysis_readme.md for interpretation guidance."