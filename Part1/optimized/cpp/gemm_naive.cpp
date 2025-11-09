
/*
C++ optimized skeleton for GEMM. Students should fill in optimizations:
- OpenMP parallelization
- Blocking / tiling
- SIMD intrinsics (AVX2 / AVX512)
- NUMA-aware allocations

Usage: ./gemm_opt N num_threads
*/
#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

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
    matmul_naive(A.data(), B.data(), C.data(), N);
    double t1 = omp_get_wtime();
    cout << "N="<<N<<" T="<<T<<" time="<<(t1-t0)<<" seconds\n";

    // simple checksum to validate
    double s = 0; for (int i=0;i<N*N;i++) s += C[i];
    cout << "checksum=" << s << "\n";
    return 0;
}
