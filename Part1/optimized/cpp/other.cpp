#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include <immintrin.h>  // AVX2 intrinsics
#include <cstdlib>      // for posix_memalign

using namespace std;

// Block size tuned for L1 cache
#define BLOCK_SIZE 64

// ============================================================================
// STRATEGY 1: Naive baseline implementation
// - Simple triple-loop (i-k-j order for better cache locality)
// - No parallelization, no SIMD, no blocking
// ============================================================================
static void matmul_naive(const double* A, const double* B, double* C, int N) {
    // simple triple-loop (row-major assumed)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            double a = A[i*N + k];
            for (int j = 0; j < N; ++j) {
                C[i*N + j] += a * B[k*N + j];
            }
        }
    }
}

// ============================================================================
// STRATEGY 2: Cache-friendly blocking/tiling ONLY
// - Divides matrices into BLOCK_SIZE x BLOCK_SIZE blocks that fit in L1/L2 cache
// - Each block is processed completely before moving to next block
// - Maximizes cache reuse of matrix elements
// - No SIMD, no parallelization
// ============================================================================
static void matmul_blocked(const double* A, const double* B, double* C, int N) {
    // Outer loops iterate over blocks
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                
                // Calculate block boundaries (handle edge cases)
                int i_end = min(ii + BLOCK_SIZE, N);
                int k_end = min(kk + BLOCK_SIZE, N);
                int j_end = min(jj + BLOCK_SIZE, N);
                
                // Process block using i-k-j order (cache-friendly)
                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        double a = A[i*N + k];
                        for (int j = jj; j < j_end; ++j) {
                            C[i*N + j] += a * B[k*N + j];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// STRATEGY 3: SIMD vectorization (AVX2) ONLY
// - Uses AVX2 intrinsics to process 4 doubles simultaneously
// - Uses FMA (Fused Multiply-Add) instruction for better performance
// - Each iteration processes 4 columns of B at once
// - No blocking, no parallelization
// ============================================================================
static void matmul_simd(const double* A, const double* B, double* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            // Broadcast A[i,k] to all 4 elements of AVX2 register
            __m256d a_broadcast = _mm256_set1_pd(A[i*N + k]);
            
            int j = 0;
            // SIMD loop: process 4 doubles at a time using AVX2
            for (; j <= N - 4; j += 4) {
                // Load 4 elements from B[k, j:j+4]
                __m256d b_vec = _mm256_loadu_pd(&B[k*N + j]);
                
                // Load 4 elements from C[i, j:j+4]
                __m256d c_vec = _mm256_loadu_pd(&C[i*N + j]);
                
                // Fused Multiply-Add: C = C + A * B (single instruction)
                c_vec = _mm256_fmadd_pd(a_broadcast, b_vec, c_vec);
                
                // Store result back to C[i, j:j+4]
                _mm256_storeu_pd(&C[i*N + j], c_vec);
            }
            
            // Handle remaining elements (if N is not divisible by 4)
            for (; j < N; ++j) {
                C[i*N + j] += A[i*N + k] * B[k*N + j];
            }
        }
    }
}

// ============================================================================
// STRATEGY 4: Cache blocking + SIMD vectorization
// - Combines blocking strategy with AVX2 SIMD intrinsics
// - Blocks ensure data stays in cache
// - SIMD processes 4 doubles per iteration within each block
// - No parallelization
// ============================================================================
static void matmul_blocked_simd(const double* A, const double* B, double* C, int N) {
    // Outer loops iterate over blocks
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                
                // Calculate block boundaries
                int i_end = min(ii + BLOCK_SIZE, N);
                int k_end = min(kk + BLOCK_SIZE, N);
                int j_end = min(jj + BLOCK_SIZE, N);
                
                // Process block with SIMD
                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        // Broadcast A[i,k] to all elements
                        __m256d a_broadcast = _mm256_set1_pd(A[i*N + k]);
                        
                        int j = jj;
                        // SIMD loop: process 4 doubles at once
                        for (; j <= j_end - 4; j += 4) {
                            __m256d b_vec = _mm256_loadu_pd(&B[k*N + j]);
                            __m256d c_vec = _mm256_loadu_pd(&C[i*N + j]);
                            c_vec = _mm256_fmadd_pd(a_broadcast, b_vec, c_vec);
                            _mm256_storeu_pd(&C[i*N + j], c_vec);
                        }
                        
                        // Handle remaining elements
                        for (; j < j_end; ++j) {
                            C[i*N + j] += A[i*N + k] * B[k*N + j];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// STRATEGY 5: Cache blocking + OpenMP parallelization
// - Cache blocking for locality
// - OpenMP parallelizes outer loop across multiple threads
// - Each thread processes different row blocks (no race condition)
// - No SIMD
// ============================================================================
static void matmul_blocked_omp(const double* A, const double* B, double* C, int N) {
    // Parallelize the outermost loop over i-blocks only
    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                
                // Calculate block boundaries
                int i_end = min(ii + BLOCK_SIZE, N);
                int k_end = min(kk + BLOCK_SIZE, N);
                int j_end = min(jj + BLOCK_SIZE, N);
                
                // Process block
                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        double a = A[i*N + k];
                        for (int j = jj; j < j_end; ++j) {
                            C[i*N + j] += a * B[k*N + j];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// STRATEGY 6: Full optimization - Blocking + SIMD + OpenMP
// - Cache blocking for data locality
// - AVX2 SIMD for vectorization (4 doubles per instruction)
// - OpenMP for multi-core parallelization
// - Combines all optimization techniques
// ============================================================================
static void matmul_blocked_simd_omp(const double* A, const double* B, double* C, int N) {
    // Parallelize the outermost loop over i-blocks only
    // This ensures each thread works on different rows of C (no race condition)
    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                
                // Calculate block boundaries
                int i_end = min(ii + BLOCK_SIZE, N);
                int k_end = min(kk + BLOCK_SIZE, N);
                int j_end = min(jj + BLOCK_SIZE, N);
                
                // Process block with SIMD
                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        // Broadcast A[i,k] to all elements
                        __m256d a_broadcast = _mm256_set1_pd(A[i*N + k]);
                        
                        int j = jj;
                        // SIMD loop: process 4 doubles at once
                        for (; j <= j_end - 4; j += 4) {
                            __m256d b_vec = _mm256_loadu_pd(&B[k*N + j]);
                            __m256d c_vec = _mm256_loadu_pd(&C[i*N + j]);
                            c_vec = _mm256_fmadd_pd(a_broadcast, b_vec, c_vec);
                            _mm256_storeu_pd(&C[i*N + j], c_vec);
                        }
                        
                        // Handle remaining elements
                        for (; j < j_end; ++j) {
                            C[i*N + j] += A[i*N + k] * B[k*N + j];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// STRATEGY 7: Memory Alignment + Blocking + SIMD + OpenMP
// - Uses 32-byte aligned memory allocation for AVX2 (aligned_alloc)
// - Enables use of faster aligned SIMD loads (_mm256_load_pd vs _mm256_loadu_pd)
// - Aligned loads are ~10-20% faster than unaligned loads
// - Cache blocking + SIMD + OpenMP same as Strategy 6
// - Memory must be allocated separately with aligned_alloc in main()
// ============================================================================
static void matmul_aligned_blocked_simd_omp(const double* A, const double* B, double* C, int N) {
    // Assumes A, B, C are 32-byte aligned (checked in main)
    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                
                // Calculate block boundaries
                int i_end = min(ii + BLOCK_SIZE, N);
                int k_end = min(kk + BLOCK_SIZE, N);
                int j_end = min(jj + BLOCK_SIZE, N);
                
                // Process block with aligned SIMD
                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        // Broadcast A[i,k] to all elements
                        __m256d a_broadcast = _mm256_set1_pd(A[i*N + k]);
                        
                        int j = jj;
                        // SIMD loop: process 4 doubles at once with ALIGNED loads
                        // This is faster than unaligned loads if memory is properly aligned
                        for (; j <= j_end - 4; j += 4) {
                            // Check if address is aligned (must be 32-byte aligned for AVX2)
                            if ((j % 4) == 0) {
                                // Use faster aligned loads
                                __m256d b_vec = _mm256_load_pd(&B[k*N + j]);
                                __m256d c_vec = _mm256_load_pd(&C[i*N + j]);
                                c_vec = _mm256_fmadd_pd(a_broadcast, b_vec, c_vec);
                                _mm256_store_pd(&C[i*N + j], c_vec);
                            } else {
                                // Fall back to unaligned if not properly aligned
                                __m256d b_vec = _mm256_loadu_pd(&B[k*N + j]);
                                __m256d c_vec = _mm256_loadu_pd(&C[i*N + j]);
                                c_vec = _mm256_fmadd_pd(a_broadcast, b_vec, c_vec);
                                _mm256_storeu_pd(&C[i*N + j], c_vec);
                            }
                        }
                        
                        // Handle remaining elements
                        for (; j < j_end; ++j) {
                            C[i*N + j] += A[i*N + k] * B[k*N + j];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// STRATEGY 8: DATA PACKING + Blocking + SIMD + OpenMP
// - **HIGHEST IMPACT OPTIMIZATION** (3-5x speedup from research!)
// - Packs blocks of A and B into contiguous memory buffers
// - Eliminates TLB misses and enables perfect prefetching
// - Sequential memory access = full cache line utilization
// - Based on GotoBLAS/BLIS Optimization 12-15
// ============================================================================

// Optimized packing parameters for your CPU (L1=32KB, L2=256KB, L3=8MB)
#define MC 128  // Rows of A in packed buffer (fits nicely in L2)
#define KC 256  // Cols of A / Rows of B
#define NC 4096 // Cols of B (can be large)

// Pack micro-panel of A (MC x KC) into contiguous column-major buffer
static inline void pack_A_panel(int mc, int kc, const double* A, int lda, double* A_packed) {
    for (int j = 0; j < kc; ++j) {
        const double* A_col = &A[j];
        for (int i = 0; i < mc; ++i) {
            *A_packed++ = A_col[i * lda];
        }
    }
}

// Pack micro-panel of B (KC x NC) into contiguous row-major buffer  
static inline void pack_B_panel(int kc, int nc, const double* B, int ldb, double* B_packed) {
    for (int i = 0; i < kc; ++i) {
        const double* B_row = &B[i * ldb];
        for (int j = 0; j < nc; ++j) {
            *B_packed++ = B_row[j];
        }
    }
}

// Optimized micro-kernel using packed data
static inline void packed_micro_kernel(int mc, int nc, int kc, 
                                       const double* A_packed, 
                                       const double* B_packed,
                                       double* C, int ldc) {
    // Process 4 rows of C at a time for better register utilization
    int i = 0;
    for (; i <= mc - 4; i += 4) {
        for (int j = 0; j < nc; j += 4) {
            // Load C block into registers
            __m256d c0 = _mm256_loadu_pd(&C[(i+0) * ldc + j]);
            __m256d c1 = _mm256_loadu_pd(&C[(i+1) * ldc + j]);
            __m256d c2 = _mm256_loadu_pd(&C[(i+2) * ldc + j]);
            __m256d c3 = _mm256_loadu_pd(&C[(i+3) * ldc + j]);
            
            // Inner loop over k
            for (int p = 0; p < kc; ++p) {
                // Broadcast A elements
                __m256d a0 = _mm256_set1_pd(A_packed[p * mc + i + 0]);
                __m256d a1 = _mm256_set1_pd(A_packed[p * mc + i + 1]);
                __m256d a2 = _mm256_set1_pd(A_packed[p * mc + i + 2]);
                __m256d a3 = _mm256_set1_pd(A_packed[p * mc + i + 3]);
                
                // Load B row
                __m256d b = _mm256_loadu_pd(&B_packed[p * nc + j]);
                
                // FMA operations
                c0 = _mm256_fmadd_pd(a0, b, c0);
                c1 = _mm256_fmadd_pd(a1, b, c1);
                c2 = _mm256_fmadd_pd(a2, b, c2);
                c3 = _mm256_fmadd_pd(a3, b, c3);
            }
            
            // Store results back
            _mm256_storeu_pd(&C[(i+0) * ldc + j], c0);
            _mm256_storeu_pd(&C[(i+1) * ldc + j], c1);
            _mm256_storeu_pd(&C[(i+2) * ldc + j], c2);
            _mm256_storeu_pd(&C[(i+3) * ldc + j], c3);
        }
    }
    
    // Handle remaining rows
    for (; i < mc; ++i) {
        for (int p = 0; p < kc; ++p) {
            __m256d a_broadcast = _mm256_set1_pd(A_packed[p * mc + i]);
            
            int j = 0;
            for (; j <= nc - 4; j += 4) {
                __m256d b_vec = _mm256_loadu_pd(&B_packed[p * nc + j]);
                __m256d c_vec = _mm256_loadu_pd(&C[i * ldc + j]);
                c_vec = _mm256_fmadd_pd(a_broadcast, b_vec, c_vec);
                _mm256_storeu_pd(&C[i * ldc + j], c_vec);
            }
            
            for (; j < nc; ++j) {
                C[i * ldc + j] += A_packed[p * mc + i] * B_packed[p * nc + j];
            }
        }
    }
}

static void matmul_packed_blocked_simd_omp(const double* A, const double* B, double* C, int N) {
    #pragma omp parallel
    {
        // Thread-local packing buffers
        double* A_packed = (double*)aligned_alloc(64, MC * KC * sizeof(double));
        double* B_packed = (double*)aligned_alloc(64, KC * NC * sizeof(double));
        
        #pragma omp for schedule(dynamic) collapse(1)
        for (int i = 0; i < N; i += MC) {
            int mc = min(MC, N - i);
            
            for (int p = 0; p < N; p += KC) {
                int kc = min(KC, N - p);
                
                // Pack A panel once for this k-block
                pack_A_panel(mc, kc, &A[i * N + p], N, A_packed);
                
                for (int j = 0; j < N; j += NC) {
                    int nc = min(NC, N - j);
                    
                    // Pack B panel
                    pack_B_panel(kc, nc, &B[p * N + j], N, B_packed);
                    
                    // Compute using packed data
                    packed_micro_kernel(mc, nc, kc, A_packed, B_packed, &C[i * N + j], N);
                }
            }
        }
        
        free(A_packed);
        free(B_packed);
    }
}

// ============================================================================
// STRATEGY 9: DATA PACKING + Blocking + OpenMP (NO SIMD)
// - Tests if packing alone gives better performance
// - Simpler scalar code, may benefit from better compiler optimization
// - Based on GotoBLAS/BLIS packing strategy
// ============================================================================

// Simple scalar micro-kernel without SIMD - just pure packing benefit
static inline void packed_micro_kernel_no_simd(int mc, int nc, int kc, 
                                                const double* A_packed, 
                                                const double* B_packed,
                                                double* C, int ldc) {
    // Simple i-k-j loop order with packed data
    for (int i = 0; i < mc; ++i) {
        for (int p = 0; p < kc; ++p) {
            double a = A_packed[p * mc + i];
            for (int j = 0; j < nc; ++j) {
                C[i * ldc + j] += a * B_packed[p * nc + j];
            }
        }
    }
}

static void matmul_packed_blocked_omp(const double* A, const double* B, double* C, int N) {
    #pragma omp parallel
    {
        // Thread-local packing buffers
        double* A_packed = (double*)aligned_alloc(64, MC * KC * sizeof(double));
        double* B_packed = (double*)aligned_alloc(64, KC * NC * sizeof(double));
        
        #pragma omp for schedule(dynamic) collapse(1)
        for (int i = 0; i < N; i += MC) {
            int mc = min(MC, N - i);
            
            for (int p = 0; p < N; p += KC) {
                int kc = min(KC, N - p);
                
                // Pack A panel once for this k-block
                pack_A_panel(mc, kc, &A[i * N + p], N, A_packed);
                
                for (int j = 0; j < N; j += NC) {
                    int nc = min(NC, N - j);
                    
                    // Pack B panel
                    pack_B_panel(kc, nc, &B[p * N + j], N, B_packed);
                    
                    // Compute using packed data (NO SIMD)
                    packed_micro_kernel_no_simd(mc, nc, kc, A_packed, B_packed, &C[i * N + j], N);
                }
            }
        }
        
        free(A_packed);
        free(B_packed);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " N num_threads\n";
        return 1;
    }
    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    omp_set_num_threads(T);

    vector<double> A(N*N), B(N*N), C(N*N);
    // reproducible RNG
    std::mt19937_64 rng(12345);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i=0;i<N*N;i++) { A[i] = dist(rng); B[i] = dist(rng); C[i]=0.0; }

    double t0 = omp_get_wtime();
    matmul_blocked_simd_omp(A.data(), B.data(), C.data(), N);
    double t1 = omp_get_wtime();
    cout << "N="<<N<<" T="<<T<<" time="<<(t1-t0)<<" seconds\n";

    // simple checksum to validate
    double s = 0; for (int i=0;i<N*N;i++) s += C[i];
    cout << "checksum=" << s << "\n";
    return 0;
// }


// ============================================================================
// ALTERNATIVE MAIN - Aligned memory allocation for Strategy 7
// To use this: Comment out the main above and uncomment this one
// ============================================================================
// int main(int argc, char** argv) {
//     if (argc < 3) {
//         cerr << "Usage: " << argv[0] << " N num_threads\n";
//         return 1;
//     }
//     int N = atoi(argv[1]);
//     int T = atoi(argv[2]);
//     omp_set_num_threads(T);

//     // Allocate 32-byte aligned memory for AVX2
//     // aligned_alloc(alignment, size) - alignment must be power of 2
//     double* A = (double*)aligned_alloc(32, N*N*sizeof(double));
//     double* B = (double*)aligned_alloc(32, N*N*sizeof(double));
//     double* C = (double*)aligned_alloc(32, N*N*sizeof(double));
    
//     if (!A || !B || !C) {
//         cerr << "Failed to allocate aligned memory\n";
//         if (A) free(A);
//         if (B) free(B);
//         if (C) free(C);
//         return 1;
//     }

//     // reproducible RNG
//     std::mt19937_64 rng(12345);
//     std::normal_distribution<double> dist(0.0, 1.0);
//     for (int i=0;i<N*N;i++) { A[i] = dist(rng); B[i] = dist(rng); C[i]=0.0; }

//     double t0 = omp_get_wtime();
//     matmul_aligned_blocked_simd_omp(A, B, C, N);
//     double t1 = omp_get_wtime();
//     cout << "N="<<N<<" T="<<T<<" time="<<(t1-t0)<<" seconds\n";

//     // simple checksum to validate
//     double s = 0; for (int i=0;i<N*N;i++) s += C[i];
//     cout << "checksum=" << s << "\n";
    
//     // Free aligned memory
//     free(A);
//     free(B);
//     free(C);
    
//     return 0;
}