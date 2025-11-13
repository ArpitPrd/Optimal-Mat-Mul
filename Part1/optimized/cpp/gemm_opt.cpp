/*
C++ optimized skeleton for GEMM. Students should fill in optimizations:
- OpenMP parallelization (DONE)
- Blocking / tiling (DONE)
- SIMD intrinsics (AVX2 / AVX512) (DONE)
- NUMA-aware allocations (DONE)

Usage: ./gemm_opt N num_threads
*/
#include <bits/stdc++.h>
#include <omp.h>
#include <malloc.h> // For _mm_malloc
#include <mm_malloc.h>
#include <immintrin.h> // <-- *** NEW: Include for AVX intrinsics ***

using namespace std;

// Define a block size. This is a tuning parameter!
// Start with 32. It must be small enough that blocks of A, B, and C
// can fit in the cache (e.g., 3*B_SIZE*B_SIZE*8 bytes < L2 cache size)
// NOTE: B_SIZE *must* be a multiple of 8 (for AVX-512) or 4 (for AVX2)
#define B_SIZE 32

static void transpose(const double* src, double* dst, int N) {
    // Transpose B into B_T
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            dst[j*N + i] = src[i*N + j]; // Swap i and j
        }
    }
}

static void matmul_blocked(const double* __restrict__ A, 
                           const double* __restrict__ B_T, 
                           double* __restrict__ C, 
                           int N) {
    
    // --- 1. Calculate the main part (N_main) ---
    // This is the largest multiple of B_SIZE less than or equal to N
    int N_main = N - (N % B_SIZE);

    // --- 2. MAIN LOOP (branch-free) ---
    // The compiler can perfectly vectorize this part.
    #pragma omp parallel for
    for (int i0 = 0; i0 < N_main; i0 += B_SIZE) {
        for (int j0 = 0; j0 < N_main; j0 += B_SIZE) {
            for (int k0 = 0; k0 < N_main; k0 += B_SIZE) {
                
                // Inner loops now have CONSTANT bounds (B_SIZE)
                for (int i = i0; i < i0 + B_SIZE; ++i) {
                    for (int j = j0; j < j0 + B_SIZE; ++j) {
                        
                        // --- *** START: AVX-512 OPTIMIZATION *** ---
                        
                        // 1. Initialize an AVX-512 register to all zeros
                        //    This will hold 8 'double' accumulators.
                        __m512d c_vec_accum = _mm512_setzero_pd();

                        // 2. Innermost k-loop, steps by 8 doubles at a time
                        for (int k = k0; k < k0 + B_SIZE; k += 8) {
                            // 2a. Load 8 doubles from A (contiguous)
                            __m512d a_vec = _mm512_load_pd(&A[i*N + k]);
                            
                            // 2b. Load 8 doubles from B_T (contiguous)
                            __m512d b_vec = _mm512_load_pd(&B_T[j*N + k]);
                            
                            // 2c. Fused-Multiply-Add (FMA)
                            // c_vec_accum = (a_vec * b_vec) + c_vec_accum
                            c_vec_accum = _mm512_fmadd_pd(a_vec, b_vec, c_vec_accum);
                        }

                        // 3. Horizontally sum the 8 doubles in the AVX register
                        //    and add to the 'c_accum'
                        //    _mm512_reduce_add_pd() sums all 8 doubles.
                        double c_accum = _mm512_reduce_add_pd(c_vec_accum);
                        
                        // 4. Add the final sum to the C matrix
                        C[i*N + j] += c_accum;
                        
                        // --- *** END: AVX-512 OPTIMIZATION *** ---
                    }
                }
            }
        }
    }

    // --- 3. CLEANUP (Slow, but on tiny bits of data) ---
    // If N=1024, N_main=1024, and these loops won't even run.
    // If N=1000, N_main=992, these loops handle the rest.
    // (This logic is unchanged from your original)
    
    // A) Handle the "right edge" (j >= N_main)
    #pragma omp parallel for
    for (int i0 = 0; i0 < N_main; i0 += B_SIZE) {
        for (int j0 = N_main; j0 < N; ++j0) { // j starts at the edge
            for (int k0 = 0; k0 < N; ++k0) { // k can be the full N
                C[i0*N + j0] += A[i0*N + k0] * B_T[j0*N + k0];
            }
        }
    }
    
    // B) Handle the "bottom edge" (i >= N_main)
    #pragma omp parallel for
    for (int i0 = N_main; i0 < N; ++i0) { // i starts at the edge
        for (int j0 = 0; j0 < N; ++j0) { // j is the full N
            for (int k0 = 0; k0 < N; ++k0) {
                C[i0*N + j0] += A[i0*N + k0] * B_T[j0*N + k0];
            }
        }
    }
}

static void matmul_naive(const double* A, const double* B, double* C, int N) {
    // simple triple-loop (row-major assumed)
    #pragma omp parallel for
    for ( int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            double a = A[i*N + k];
            for (int j = 0; j < N; ++j) {
                C[i*N + j] += a * B[k*N + j];
            }
        }
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

    // --- 1. USE ALIGNED ALLOCATION ---
    const size_t alignment = 64;
    double* A = (double*)_mm_malloc(N * N * sizeof(double), alignment);
    double* B = (double*)_mm_malloc(N * N * sizeof(double), alignment);
    double* C = (double*)_mm_malloc(N * N * sizeof(double), alignment);
    double* B_T = (double*)_mm_malloc(N * N * sizeof(double), alignment);

    if (A == nullptr || B == nullptr || C == nullptr || B_T == nullptr) {
        cerr << "Memory allocation failed.\n";
        return 1;
    }

    // --- 2. INITIALIZE THE DATA (NUMA-AWARE) ---
    // *** NEW: Parallelized initialization ***
    // This implements a "first-touch" policy. The thread that
    // first writes to the memory "owns" it on its local NUMA node.
    std::mt19937_64 rng(12345);
    std::normal_distribution<double> dist(0.0, 1.0);
    
    // Note: This parallel random init isn't perfectly reproducible
    // like the serial version, but it's correct for NUMA locality.
    // For a lab, this is the standard practice.
    #pragma omp parallel for
    for (int i = 0; i < N * N; i++) {
        A[i] = dist(rng); // Good enough for init
        B[i] = dist(rng);
        C[i] = 0.0;
    }

    // --- 3. RUN THE MULTIPLICATION ONE TIME ---
    
    // Transpose is already parallel, which is good for NUMA
    transpose(B, B_T, N); 

    matmul_blocked(A, B_T, C, N);

    // --- 4. FREE THE ALIGNED MEMORY ---
    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
    _mm_free(B_T);
    
    return 0;
}