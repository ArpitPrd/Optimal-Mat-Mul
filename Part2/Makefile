# ===========================
#  Smith-Waterman Build System
# ===========================

CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra

# Baseline paths
BASELINE_SRC = baseline/sw_baseline.c
BASELINE_BIN = baseline/bin/sw_baseline

# Optimized paths
OPT_SRC = optimized/sw_opt.c
OPT_BIN = optimized/bin/sw_opt

# Default target
all: $(BASELINE_BIN) $(OPT_BIN)

# Create bin directories if not present
baseline/bin:
	mkdir -p baseline/bin

optimized/bin:
	mkdir -p optimized/bin

# Build baseline version
$(BASELINE_BIN): $(BASELINE_SRC) | baseline/bin
	$(CC) $(CFLAGS) $(BASELINE_SRC) -o $(BASELINE_BIN)

# Build optimized version INCLUDING SIMD flags
$(OPT_BIN): $(OPT_SRC) | optimized/bin
	$(CC) $(CFLAGS) -msse4.1 -mavx2 -mavx512bw $(OPT_SRC) -o $(OPT_BIN)

# Clean all builds
clean:
	rm -rf baseline/bin optimized/bin

.PHONY: all clean