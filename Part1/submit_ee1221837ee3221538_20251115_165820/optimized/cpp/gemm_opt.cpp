#include <bits/stdc++.h>
#include <immintrin.h>
#include <omp.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/sysinfo.h>
#include <sched.h>

#ifdef HAS_NUMA
#include <numa.h>
#include <numaif.h>
#endif

using namespace std;

static bool g_numa_available = false; // Will remain false if HAS_NUMA is not defined

static void check_numa_support() {
#ifdef HAS_NUMA
    // This code block only exists if -DHAS_NUMA was passed
    if (numa_available() == 0) {
        g_numa_available = true;
        cout << "NUMA support detected: " << numa_num_configured_nodes() << " nodes." << endl;
        cout << "Info: Using parallel first-touch policy for NUMA-aware allocation." << endl;
    } else {
        cout << "NUMA support not available (libnuma error)." << endl;
    }
#else
    // This code block exists if -DHAS_NUMA was *not* passed
    cout << "NUMA support not compiled. (Header 'numa.h' not found during build)." << endl;
#endif
}

// ---------- Feature detection ----------
static inline bool has_avx2() {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}
static inline bool has_avx512f() {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx512f");
#else
    return false;
#endif
}
static inline bool has_fma() {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

// aligned malloc wrapper
static void* aligned_malloc(size_t bytes, size_t align) {
    void* ptr = nullptr;
#if defined(_ISOC11_SOURCE)
    ptr = aligned_alloc(align, ((bytes + align -1)/align)*align);
    if (!ptr) { /* fallthrough to posix */ }
#endif
    if (!ptr) {
        if (posix_memalign(&ptr, align, bytes) != 0) ptr = nullptr;
    }
    return ptr;
}
static void aligned_free(void* p) { free(p); }

/**
 * @brief Transposes matrix B into B_T in parallel.
 * This is crucial for NUMA first-touch on B_T.
 */
static void transpose(const double* B, double* B_T, int N) {
    const int TB = 64; // Tiling block size
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i += TB) {
        int i_max = min(N, i + TB);
        for (int j = 0; j < N; j += TB) {
            int j_max = min(N, j + TB);
            for (int ii = i; ii < i_max; ++ii) {
                for (int jj = j; jj < j_max; ++jj) {
                    B_T[jj*N + ii] = B[ii*N + jj]; // First touch on B_T[jj*N + ii]
                }
            }
        }
    }
}

// Generic blocked matmul (fallback)
static void gemm_blocked_generic(const double* A, const double* B_T, double* C, int N,
                                 int MC, int KC, int NC) {
    #pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 < N; i0 += MC) {
        int ib = min(MC, N - i0);
        for (int k0 = 0; k0 < N; k0 += KC) {
            int kb = min(KC, N - k0);
            for (int j0 = 0; j0 < N; j0 += NC) {
                int jb = min(NC, N - j0);
                for (int i = i0; i < i0 + ib; ++i) {
                    for (int k = k0; k < k0 + kb; ++k) {
                        double a = A[i*N + k];
                        double* crow = &C[i*N + j0];
                        for (int j = 0; j < jb; ++j) {
                            crow[j] += a * B_T[(j0 + j)*N + k];
                        }
                    }
                }
            }
        }
    }
}

// -------- AVX2 kernel (function-level target) --------
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
static void gemm_blocked_avx2_kernel(const double* A, const double* B_T, double* C, int N, int i_start, int i_end, int k_start, int k_end, int j0, int jb) {
    const bool use_fma = has_fma(); 
    const int V = 4; // 4 doubles in 256-bit
    for (int i = i_start; i < i_end; ++i) {
        for (int k = k_start; k < k_end; ++k) {
            double a = A[i*N + k];
            __m256d va = _mm256_set1_pd(a);
            int j = j0;
            for (; j + V - 1 < j0 + jb; j += V) {
                __m256d vc = _mm256_loadu_pd(&C[i*N + j]);
                __m256d vb = _mm256_loadu_pd(&B_T[j*N + k]);
                if (use_fma) {
                    vc = _mm256_fmadd_pd(va, vb, vc);
                } else {
                    vc = _mm256_add_pd(vc, _mm256_mul_pd(va, vb));
                }
                _mm256_storeu_pd(&C[i*N + j], vc);
            }
            for (; j < j0 + jb; ++j) {
                C[i*N + j] += a * B_T[j*N + k];
            }
        }
    }
}

static void gemm_blocked_avx2(const double* A, const double* B_T, double* C, int N, int MC, int KC, int NC) {
    
    #pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 < N; i0 += MC) {
        int ib = min(MC, N - i0);
        for (int k0 = 0; k0 < N; k0 += KC) {
            int kb = min(KC, N - k0);
            for (int j0 = 0; j0 < N; j0 += NC) {
                int jb = min(NC, N - j0);
                for (int i = i0; i < i0 + ib; ++i) {
                    gemm_blocked_avx2_kernel(A, B_T, C, N, i, i+1, k0, k0+kb, j0, jb);
                }
            }
        }
    }
}

// -------- AVX-512 kernel (function-level target) --------
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f")))
#endif
static void gemm_blocked_avx512_kernel(const double* A, const double* B_T, double* C, int N, int i_start, int i_end, int k_start, int k_end, int j0, int jb) {
    const int V = 8; // 8 doubles in 512-bit
    for (int i = i_start; i < i_end; ++i) {
        for (int k = k_start; k < k_end; ++k) {
            double a = A[i*N + k];
            __m512d va = _mm512_set1_pd(a);
            int j = j0;
            for (; j + V - 1 < j0 + jb; j += V) {
                __m512d vc = _mm512_loadu_pd(&C[i*N + j]);
                __m512d vb = _mm512_loadu_pd(&B_T[j*N + k]);
                vc = _mm512_fmadd_pd(va, vb, vc);
                _mm512_storeu_pd(&C[i*N + j], vc);
            }
            for (; j < j0 + jb; ++j) {
                C[i*N + j] += a * B_T[j*N + k];
            }
        }
    }
}

static void gemm_blocked_avx512(const double* A, const double* B_T, double* C, int N, int MC, int KC, int NC) {
    
    #pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 < N; i0 += MC) {
        int ib = min(MC, N - i0);
        for (int k0 = 0; k0 < N; k0 += KC) {
            int kb = min(KC, N - k0);
            for (int j0 = 0; j0 < N; j0 += NC) {
                int jb = min(NC, N - j0);
                for (int i = i0; i < i0 + ib; ++i) {
                    gemm_blocked_avx512_kernel(A,B_T,C,N,i,i+1,k0,k0+kb,j0,jb);
                }
            }
        }
    }
}

// High-level driver
static void gemm_driver(const double* A, const double* B, double* C, int N, int threads) {
    double* B_T = (double*)aligned_malloc(sizeof(double)*(size_t)N*(size_t)N, 64);
    if (!B_T) { cerr << "Allocation failed\n"; exit(1); }
    
    omp_set_num_threads(threads);
    omp_set_dynamic(0);
    omp_set_schedule(omp_sched_static, 0);

    // Call the now-parallel transpose.
    // This provides first-touch benefits even on non-NUMA systems
    // by improving cache locality.
    transpose(B, B_T, N);

    int MC = 128;
    int KC = 256;
    int NC = 256;
    MC = min(MC, N); KC = min(KC, N); NC = min(NC, N);

    const char* force = getenv("KERNEL");
    bool used = false;
    if (force) {
        string f(force);
        if (f=="avx512" && has_avx512f()) { gemm_blocked_avx512(A,B_T,C,N,MC,KC,NC); used = true; }
        else if (f=="avx2" && has_avx2()) { gemm_blocked_avx2(A,B_T,C,N,MC,KC,NC); used = true; }
        else if (f=="generic") { gemm_blocked_generic(A,B_T,C,N,MC,KC,NC); used = true; }
    }
    if (!used) {
        if (has_avx512f()) gemm_blocked_avx512(A,B_T,C,N,MC,KC,NC);
        else if (has_avx2()) gemm_blocked_avx2(A,B_T,C,N,MC,KC,NC);
        else gemm_blocked_generic(A,B_T,C,N,MC,KC,NC);
    }
    aligned_free(B_T);
}

// ---------- main ----------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " N num_threads\n";
        return 1;
    }
    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    if (N <= 0 || T <= 0) { cerr << "Invalid args\n"; return 1; }

    cout << "Threads requested: " << T << "\n";
    omp_set_num_threads(T);
    setenv("OMP_PROC_BIND", "TRUE", 1);
    setenv("OMP_PLACES", "cores", 1);

    // This call is now safe, whether compiled with NUMA or not.
    check_numa_support();

    cout << "Detected: AVX2=" << (has_avx2() ? "yes" : "no")
         << " AVX-512=" << (has_avx512f() ? "yes" : "no")
         << " FMA=" << (has_fma() ? "yes" : "no") << "\n";

    size_t bytes = (size_t)N * N * sizeof(double);
    double* A = (double*)aligned_malloc(bytes, 64);
    double* B = (double*)aligned_malloc(bytes, 64);
    double* C = (double*)aligned_malloc(bytes, 64);
    if (!A || !B || !C) { cerr << "OOM\n"; return 1; }

    std::mt19937_64 rng(1234567);
    vector<std::mt19937_64> thread_rngs;
    thread_rngs.reserve(T);
    for (int i = 0; i < T; ++i) {
        thread_rngs.emplace_back(rng());
    }
    
    cout << "Initializing matrices in parallel (first-touch policy)..." << endl;
    
    // This parallel initialization is *always* good, even without NUMA,
    // as it establishes cache affinity.
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::normal_distribution<double> local_dist(0.0, 1.0);
        std::mt19937_64& local_rng = thread_rngs[tid];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < (size_t)N * N; ++i) {
            A[i] = local_dist(local_rng);
            B[i] = local_dist(local_rng);
            C[i] = 0.0; // Crucial first-touch on C
        }
    }
    cout << "Initialization complete." << endl;

    double t0 = omp_get_wtime();
    gemm_driver(A,B,C,N,T);
    double t1 = omp_get_wtime();

    double s = 0.0;
    #pragma omp parallel for reduction(+:s)
    for (size_t i = 0; i < (size_t)N * N; ++i) s += C[i];

    cout.setf(std::ios::fixed); cout<<setprecision(6);
    cout << "N="<<N<<" T="<<T<<" time="<<(t1 - t0)<<" s\n";
    double gflops = (2.0 * (double)N * N * N) / ((t1 - t0) * 1e9);
    cout << "GFLOPS=" << gflops << " checksum=" << s << "\n";

    aligned_free(A); aligned_free(B); aligned_free(C);
    return 0;
}