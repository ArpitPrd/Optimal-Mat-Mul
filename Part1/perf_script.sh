#!/usr/bin/env bash
set -e

# This script runs targeted 'perf stat' commands to gather profiling
# evidence for the lab report.
#
# It specifically tests the hypotheses for the "weird" N=512 curve
# and the "memory-bound" N=2048 curve.
#
# VM-SAFE UPDATE: We now explicitly specify all events using '-e'
# to avoid requesting unsupported default events (like 'slots')
# which often fail inside virtual machines.

BIN_FILE="optimized/gemm_opt"

if [ ! -x "$BIN_FILE" ]; then
    echo "Error: Optimized binary '$BIN_FILE' not found." >&2
    echo "Please compile it first (e.g., using run.sh)" >&2
    exit 1
fi

# Define a common set of events that gives us GHz and IPC
# 'task-clock' and 'cycles' are needed to calculate GHz
# 'instructions' and 'cycles' are needed to calculate IPC
COMMON_EVENTS="task-clock,cycles,instructions"

echo "========================================================================"
echo "Hypothesis 1: Analyzing N=512 (AVX-512 Frequency Scaling)"
echo "We are checking if the 'GHz' (clock speed) dips at 8 threads."
echo "========================================================================"

echo "--- Profiling N=512, T=4 (The 'sweet spot') ---"
# -r 3 runs the command 3 times and averages the results for stability
# FIX: Added -e $COMMON_EVENTS
perf stat -e $COMMON_EVENTS -r 3 $BIN_FILE 512 4
echo "------------------------------------------------"

echo "--- Profiling N=512, T=8 (The 'dip') ---"
# FIX: Added -e $COMMON_EVENTS
perf stat -e $COMMON_EVENTS -r 3 $BIN_FILE 512 8
echo "------------------------------------------------"

echo "--- Profiling N=512, T=16 (The 'recovery') ---"
# FIX: Added -e $COMMON_EVENTS
perf stat -e $COMMON_EVENTS -r 3 $BIN_FILE 512 16
echo "------------------------------------------------"


echo ""
echo "========================================================================"
echo "Hypothesis 2: Analyzing N=2048 (Memory-Bound)"
echo "We are checking for low IPC and high cache misses."
echo "========================================================================"

# FIX: Added 'task-clock' to the existing list to ensure we get GHz here too.
# These events are generally safe in VMs.
VM_MEM_EVENTS="task-clock,cycles,instructions,cache-misses,L1-dcache-load-misses,LLC-load-misses"

echo "--- Profiling N=2048, T=1 (Baseline) ---"
perf stat -e $VM_MEM_EVENTS -r 3 $BIN_FILE 2048 1
echo "------------------------------------------------"

echo "--- Profiling N=2048, T=16 (Scaled) ---"
perf stat -e $VM_MEM_EVENTS -r 3 $BIN_FILE 2048 16
echo "------------------------------------------------"

echo "Profiling complete. See analysis_readme.md for interpretation guidance."