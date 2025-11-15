#!/usr/bin/env python3
"""
Auto-scorer for Smith-Waterman project.
Works with folder structure:

baseline/bin/sw_baseline
optimized/bin/sw_opt

Usage:
    ./auto_score.py baseline 1000
    ./auto_score.py optimized 1000
    ./auto_score.py both 1000

This script:
  1. runs `make` to build both versions
  2. runs baseline and/or optimized with the given integer argument
  3. measures runtime
  4. prints speedup if both are selected
"""

import argparse
import subprocess
import time
import sys
import os

# --------------------------------------
# Argument Parser
# --------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("mode",
                    choices=["baseline", "optimized", "both"],
                    help="Which version to run")
parser.add_argument("arg", type=str,
                    help="Integer argument to pass to executables")
parser.add_argument("--reps", type=int, default=3,
                    help="Number of repetitions per run")
args = parser.parse_args()


# --------------------------------------
# Helper: run a shell command and time it
# --------------------------------------
def run_cmd(cmd):
    t0 = time.time()
    proc = subprocess.run(cmd,
                          shell=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          universal_newlines=True)
    t1 = time.time()

    if proc.returncode != 0:
        print("Command failed:", cmd)
        print(proc.stdout)
        print(proc.stderr)
        sys.exit(1)

    return t1 - t0, proc.stdout


# --------------------------------------
# Ensure executables exist → build with make
# --------------------------------------
print("== Building with Makefile ==")
subprocess.run("make", shell=True, check=True)

BASE_EXE = "./baseline/bin/sw_baseline"
OPT_EXE = "./optimized/bin/sw_opt"

if not os.path.isfile(BASE_EXE) or not os.path.isfile(OPT_EXE):
    print("Error: executables not found after building.")
    sys.exit(1)

# --------------------------------------
# RUN BASELINE
# --------------------------------------
if args.mode in ["baseline", "both"]:
    cmd = f"{BASE_EXE} {args.arg}"
    print("\n== Running baseline ==")
    base_times = []
    for i in range(args.reps):
        t, out = run_cmd(cmd)
        print(out)
        base_times.append(t)
    base_med = sorted(base_times)[len(base_times)//2]
    print(f"Baseline median runtime: {base_med:.6f}s")
else:
    base_med = None

# --------------------------------------
# RUN OPTIMIZED
# --------------------------------------
if args.mode in ["optimized", "both"]:
    cmd = f"{OPT_EXE} {args.arg}"
    print("\n== Running optimized ==")
    opt_times = []
    for i in range(args.reps):
        t, out = run_cmd(cmd)
        print(out)
        opt_times.append(t)
    opt_med = sorted(opt_times)[len(opt_times)//2]
    print(f"Optimized median runtime: {opt_med:.6f}s")
else:
    opt_med = None

# --------------------------------------
# PRINT SPEEDUP (if running both)
# --------------------------------------
if args.mode == "both":
    speedup = base_med / opt_med
    print("\n== Speedup Summary ==")
    print(f"Baseline : {base_med:.6f}s")
    print(f"Optimized: {opt_med:.6f}s")
    print(f"Speedup  : {speedup:.3f}x")
