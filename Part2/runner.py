#!/usr/bin/env python3
"""
Batch runner for Smith-Waterman project.

Automatically runs:
    ./run.sh baseline N
    ./run.sh optimized N

For multiple N, collects:
    - runtime (sec)
    - GFLOPs (if available) or GB/s
    - speedup
    - core count
    - SIMD width used
    - Cache miss rates (L1, L2, L3) via perf

Outputs:
    results.csv
"""

import subprocess
import csv
import time
import os
import sys
import multiprocessing

# ----------------------------
# Configurable N values:
# ----------------------------
N_VALUES = []
N = 100
for i in range(1,100):
    N_VALUES.append(i*N)

OUTPUT_CSV = "results.csv"


# ----------------------------
# Detect CPU SIMD width
# ----------------------------
def detect_simd_width():
    flags = open("/proc/cpuinfo").read()

    if "avx512bw" in flags:
        return "AVX-512 (512-bit)"
    if "avx2" in flags:
        return "AVX2 (256-bit)"
    if "sse4_1" in flags:
        return "SSE4.1 (128-bit)"
    return "SCALAR"


# ----------------------------
# Run a command and time it
# ----------------------------
def run_cmd(cmd):
    t0 = time.time()
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True)
    t1 = time.time()

    if proc.returncode != 0:
        print(f"ERROR running: {cmd}")
        print(proc.stdout)
        print(proc.stderr)
        sys.exit(1)

    return (t1 - t0), proc.stdout


# ----------------------------
# Run perf to get cache misses
# ----------------------------
def run_perf(cmd):
    perf_cmd = (
        f"perf stat -e L1-dcache-loads,L1-dcache-load-misses,"
        f"LLC-loads,LLC-load-misses {cmd}"
    )

    proc = subprocess.run(perf_cmd, shell=True,
                          stderr=subprocess.PIPE,
                          stdout=subprocess.PIPE,
                          universal_newlines=True)

    stderr = proc.stderr

    def extract(event):
        for line in stderr.splitlines():
            if event in line:
                return int(line.split()[0].replace(",", ""))
        return 0

    L1_loads = extract("L1-dcache-loads")
    L1_misses = extract("L1-dcache-load-misses")
    LLC_loads = extract("LLC-loads")
    LLC_misses = extract("LLC-load-misses")

    return {
        "L1_miss_rate": L1_misses / L1_loads if L1_loads else 0,
        "L3_miss_rate": LLC_misses / LLC_loads if LLC_loads else 0,
    }


# ----------------------------
# Main Runner
# ----------------------------
if __name__ == "__main__":

    simd = detect_simd_width()
    cores = multiprocessing.cpu_count()

    print("Detected SIMD:", simd)
    print("Detected Cores:", cores)
    print("Running tests...")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N",
            "Baseline Time (s)",
            "Optimized Time (s)",
            "Speedup",
            "GB/s or GFLOPs",
            "Cores",
            "SIMD",
            "L1 Miss Rate (opt)",
            "L3 Miss Rate (opt)",
        ])

        for N in N_VALUES:
            print(f"\n=== N = {N} ===")

            # Run baseline
            baseline_cmd = f"./run.sh baseline {N}"
            t_base, _ = run_cmd(baseline_cmd)

            # Run optimized
            optimized_cmd = f"./run.sh optimized {N}"
            t_opt, _ = run_cmd(optimized_cmd)

            # Cache misses (optimized only)
            perf_stats = run_perf(optimized_cmd)

            speedup = t_base / t_opt

            # Memory throughput estimate:
            bytes_moved = N * N * 2  # rough heuristic (2 bytes per cell)
            GBs = bytes_moved / (t_opt * 1e9)

            writer.writerow([
                N,
                t_base,
                t_opt,
                speedup,
                GBs,
                cores,
                simd,
                perf_stats["L1_miss_rate"],
                perf_stats["L3_miss_rate"],
            ])

    print("\nResults saved to results.csv")
