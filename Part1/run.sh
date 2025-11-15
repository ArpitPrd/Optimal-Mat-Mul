#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run.sh <N> <T> <output_csv_file>

# --- Configuration ---
N=${1}
T=${2}
OUT_FILE=${3}

SRC_FILE_OPT="optimized/cpp/gemm_opt.cpp"
BIN_FILE_OPT="optimized/gemm_opt"
BIN_FILE_BASE="baseline/gemm_baseline.py"
NUM_SAMPLES=3

# --- Validation ---
if [ -z "$N" ] || [ -z "$T" ] || [ -z "$OUT_FILE" ]; then
    echo "Usage: $0 <N> <T> <output_csv_file>"
    exit 1
fi

# --- 1. Compile With NUMA Detection ---
NUMA_CFLAGS=""
NUMA_LIBS=""

echo "#include <numa.h>" > .numa_check.cpp
if g++ -E .numa_check.cpp -o /dev/null &> /dev/null; then
    NUMA_CFLAGS="-DHAS_NUMA"
    NUMA_LIBS="-lnuma"
else
    echo "--- NUMA headers not found ---"
fi
rm -f .numa_check.cpp

echo "Compiling optimized GEMM..."
g++ -O3 -march=native -fopenmp -std=c++17 $NUMA_CFLAGS $SRC_FILE_OPT -o $BIN_FILE_OPT $NUMA_LIBS

# --- 2. Warmup ---
echo "Warmup run..."
$BIN_FILE_OPT $N $T > /dev/null

# --- 3. Optimized C++ Runs ---
total_time_opt=0
total_gflops_opt=0

for i in $(seq 1 $NUM_SAMPLES); do
    OUT=$($BIN_FILE_OPT $N $T)
    DATA=$(echo "$OUT" | grep "^N=")
    GF=$(echo "$OUT" | grep "^GFLOPS=")

    TIME_VAL=$(echo "$DATA" | awk -F'[= ]' '{print $6}')
    GFLOPS_VAL=$(echo "$GF" | awk -F'[= ]' '{print $2}')

    total_time_opt=$(echo "$total_time_opt + $TIME_VAL" | bc)
    total_gflops_opt=$(echo "$total_gflops_opt + $GFLOPS_VAL" | bc)
done

avg_time_opt=$(echo "scale=6; $total_time_opt / $NUM_SAMPLES" | bc)
avg_gflops_opt=$(echo "scale=6; $total_gflops_opt / $NUM_SAMPLES" | bc)

echo "Optimized Avg: Time=$avg_time_opt  GFLOPS=$avg_gflops_opt"

# --- 4. Baseline Python Runs ---
echo "Running baseline Python..."
( /usr/bin/time -p python3 $BIN_FILE_BASE $N $T ) &>/dev/null

total_time_base=0
for i in $(seq 1 $NUM_SAMPLES); do
    t=$( { /usr/bin/time -p python3 $BIN_FILE_BASE $N $T 2>&1; } | grep real | awk '{print $2}')
    total_time_base=$(echo "$total_time_base + $t" | bc)
done

avg_time_base=$(echo "scale=6; $total_time_base / $NUM_SAMPLES" | bc)
echo "Baseline Avg Time = $avg_time_base"

speedup=$(echo "scale=6; $avg_time_base / $avg_time_opt" | bc)

# --- 5. PERF Collection (Optimized Only) ---
PERF_FILE="perf_tmp_${N}_${T}.txt"

perf stat -o $PERF_FILE \
    -e cycles,instructions,branches,branch-misses,\
cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,\
L1-icache-loads,L1-icache-load-misses,\
LLC-loads,LLC-load-misses \
    $BIN_FILE_OPT $N $T >/dev/null 2>&1

extract() {
    grep "$1" $PERF_FILE | awk '{print $1}' | tr -d ','
}

cycles=$(extract "cycles")
instr=$(extract "instructions")
branches=$(extract "branches ")
br_miss=$(extract "branch-misses")
l1_load=$(extract "L1-dcache-loads")
l1_miss=$(extract "L1-dcache-load-misses")
llc_load=$(extract "LLC-loads")
llc_miss=$(extract "LLC-load-misses")

ipc=$(echo "scale=6; $instr / $cycles" | bc)

l1_miss_rate=$(echo "scale=6; $l1_miss / $l1_load" | bc)
llc_miss_rate=$(echo "scale=6; $llc_miss / $llc_load" | bc)
branch_miss_rate=$(echo "scale=6; $br_miss / $branches" | bc)

# --- 6. Append to CSV ---
echo "$N,$T,$avg_time_opt,$avg_gflops_opt,$avg_time_base,$speedup,$cycles,$instr,$ipc,$l1_load,$l1_miss,$l1_miss_rate,$llc_load,$llc_miss,$llc_miss_rate,$branches,$br_miss,$branch_miss_rate" >> $OUT_FILE

echo "Logged results into $OUT_FILE"
