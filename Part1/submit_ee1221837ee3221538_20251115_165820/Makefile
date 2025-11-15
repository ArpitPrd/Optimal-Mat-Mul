CXX = g++
# Base flags
BASE_CXXFLAGS = -O3 -march=native -fopenmp -DNDEBUG -std=c++17
BASE_LDFLAGS = -fopenmp

# --- NUMA Auto-detection ---
# We use shell commands to check for numa.h, similar to the run.sh script.
# The result is stored in Makefile variables.

# Check for numa.h availability.
# We create a temp file, try to preprocess it, and check the exit code.
# The '@' suppresses the command from being printed.
# We then clean up the temp file.
# The result (compiler flags) is stored in NUMA_CFLAGS.
NUMA_CHECK_FILE = .numa_check.cpp
NUMA_CFLAGS = $(shell echo "#include <numa.h>" > $(NUMA_CHECK_FILE); \
                  if $(CXX) -E $(NUMA_CHECK_FILE) -o /dev/null >/dev/null 2>&1; then \
                      echo "-DHAS_NUMA"; \
                  else \
                      echo ""; \
                  fi; \
                  rm -f $(NUMA_CHECK_FILE))

# Set linker flags based on whether NUMA_CFLAGS was set.
ifeq ($(NUMA_CFLAGS), -DHAS_NUMA)
    NUMA_LDFLAGS = -lnuma
    INFO_MSG = "--- Compiling with NUMA-aware optimizations (-DHAS_NUMA, -lnuma). ---"
else
    NUMA_LDFLAGS =
    INFO_MSG = "--- Compiling without NUMA-aware optimizations (numa.h not found). ---"
endif
# --- End NUMA Auto-detection ---

# Combine base flags with NUMA flags
CXXFLAGS = $(BASE_CXXFLAGS) $(NUMA_CFLAGS)
LDFLAGS = $(BASE_LDFLAGS) $(NUMA_LDFLAGS)

all: optimized/gemm_opt

optimized/gemm_opt: optimized/cpp/gemm_opt.cpp
	@echo "$INFO_MSG"
	@mkdir -p optimized
	$(CXX) $(CXXFLAGS) -o optimized/gemm_opt optimized/cpp/gemm_opt.cpp $(LDFLAGS)
	@echo "Build successful."

clean:
	rm -f optimized/gemm_opt $(NUMA_CHECK_FILE)

.PHONY: all clean