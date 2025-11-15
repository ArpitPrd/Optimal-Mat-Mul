#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

#define MAX(A,B) ((A) > (B) ? (A) : (B))
#define MATCH     2
#define MISMATCH -1
#define GAP      -2

/* ============================================================
    *  MULTI-ISA SMITH–WATERMAN DISPATCH (SSE → AVX2 → AVX512)
    *  Pure C, safe loads, 2-row DP, SIMD accelerated where possible
    * ============================================================ */

/* ---------- Scalar fallback ---------- */
static int sw_scalar(const char *a, const char *b, int n, int m) {
    int *prev = calloc(m+1, sizeof(int));
    int *curr = calloc(m+1, sizeof(int));
    int maxv = 0;

    for (int i = 1; i <= n; i++) {
        curr[0] = 0;
        for (int j = 1; j <= m; j++) {
            int diag = prev[j-1] + (a[i-1] == b[j-1] ? MATCH : MISMATCH);
            int up   = prev[j]   + GAP;
            int left = curr[j-1] + GAP;

            int best = diag;
            if (up   > best) best = up;
            if (left > best) best = left;
            if (best < 0)    best = 0;

            curr[j] = best;
            if (best > maxv) maxv = best;
        }
        int *t = prev; prev = curr; curr = t;
    }

    free(prev); free(curr);
    return maxv;
}

    /* ============================================================
     *              SSE SIMD VERSION  (8 lanes of int16)
     * ============================================================ */
#ifdef __SSE4_1__
    static int sw_sse41_full(const char *a, const char *b, int n, int m)
{

    /* Allocate full DP matrix H[(n+1) × (m+1)] aligned to 16 bytes */
    int16_t **H = (int16_t**)malloc((n+1) * sizeof(int16_t*));
    for (int i = 0; i <= n; i++) {
        void *p = NULL;
        posix_memalign(&p, 16, (m+1) * sizeof(int16_t));  // SSE wants 16-byte alignment
        H[i] = (int16_t*)p;
        memset(H[i], 0, (m+1) * sizeof(int16_t));
    }

    __m128i vGap  = _mm_set1_epi16(GAP);
    __m128i vZero = _mm_set1_epi16(0);

    int maxv = 0;

    /* MAIN DP LOOP */
    for (int i = 1; i <= n; i++) {

        for (int j = 1; j <= m; j += 8) {

            int rem = m - j + 1;
            int VL = (rem >= 8 ? 8 : rem);

            int16_t A[8], diagBuf[8], upBuf[8], leftBuf[8];

            /* MATCH/MISMATCH vector */
            for (int k = 0; k < VL; k++)
                A[k] = (a[i-1] == b[j-1+k]) ? MATCH : MISMATCH;
            for (int k = VL; k < 8; k++)
                A[k] = 0;

            __m128i vMat = _mm_load_si128((__m128i*)A);

            /* H[i-1][j-1..j+6] → diag */
            for (int k = 0; k < VL; k++)
                diagBuf[k] = H[i-1][j-1+k];
            for (int k = VL; k < 8; k++)
                diagBuf[k] = 0;
            __m128i vDiagBase = _mm_load_si128((__m128i*)diagBuf);

            /* H[i-1][j..j+7] → up */
            for (int k = 0; k < VL; k++)
                upBuf[k] = H[i-1][j+k];
            for (int k = VL; k < 8; k++)
                upBuf[k] = 0;
            __m128i vUpBase = _mm_load_si128((__m128i*)upBuf);

            /* H[i][j-1..j+6] → left */
            for (int k = 0; k < VL; k++)
                leftBuf[k] = H[i][j-1+k];
            for (int k = VL; k < 8; k++)
                leftBuf[k] = 0;
            __m128i vLeftBase = _mm_load_si128((__m128i*)leftBuf);

            /* DP operations */
            __m128i vDiag = _mm_adds_epi16(vDiagBase, vMat);
            __m128i vUp   = _mm_adds_epi16(vUpBase,   vGap);
            __m128i vLf   = _mm_adds_epi16(vLeftBase, vGap);

            __m128i vMax1 = _mm_max_epi16(vDiag, vUp);
            __m128i vMax2 = _mm_max_epi16(vLf,  vMax1);
            __m128i vRes  = _mm_max_epi16(vZero, vMax2);

            /* Write results back */
            int16_t tmp[8];
            _mm_store_si128((__m128i*)tmp, vRes);

            for (int k = 0; k < VL; k++) {
                H[i][j+k] = tmp[k];
                if (tmp[k] > maxv) maxv = tmp[k];
            }
        }
    }

    /* free */
    for (int i = 0; i <= n; i++) free(H[i]);
    free(H);

    return maxv;
}

#endif

    /* ============================================================
     *              AVX2 SIMD VERSION (16 lanes of int16)
     * ============================================================ */
#ifdef __AVX2__

    static int sw_avx2_full(const char *a, const char *b, int n, int m)
{

    /* --- Allocate full DP matrix H[(n+1) × (m+1)] --- */
    int16_t **H = (int16_t**)malloc((n+1) * sizeof(int16_t*));
    for (int i = 0; i <= n; i++) {
        void *p = NULL;
        posix_memalign(&p, 32, (m+1) * sizeof(int16_t));  // AVX2 needs 32-byte alignment
        H[i] = (int16_t*)p;
        memset(H[i], 0, (m+1) * sizeof(int16_t));
    }

    __m256i vGap  = _mm256_set1_epi16(GAP);
    __m256i vZero = _mm256_set1_epi16(0);

    int maxv = 0;

    /* --- MAIN DP --- */
    for (int i = 1; i <= n; i++) {

        for (int j = 1; j <= m; j += 16) {

            int rem = m - j + 1;
            int VL = rem >= 16 ? 16 : rem;

            int16_t A[16], diagBuf[16], upBuf[16], leftBuf[16];

            /* MATCH/MISMATCH vector */
            for (int k = 0; k < VL; k++)
                A[k] = (a[i-1] == b[j-1+k]) ? MATCH : MISMATCH;
            for (int k = VL; k < 16; k++)
                A[k] = 0;

            __m256i vMat = _mm256_load_si256((__m256i*)A);

            /* Load H[i-1][j-1 .. j+14] → diag */
            for (int k = 0; k < VL; k++)
                diagBuf[k] = H[i-1][j-1+k];
            for (int k = VL; k < 16; k++)
                diagBuf[k] = 0;
            __m256i vDiagBase = _mm256_load_si256((__m256i*)diagBuf);

            /* Load H[i-1][j .. j+15] → up */
            for (int k = 0; k < VL; k++)
                upBuf[k] = H[i-1][j+k];
            for (int k = VL; k < 16; k++)
                upBuf[k] = 0;
            __m256i vUpBase = _mm256_load_si256((__m256i*)upBuf);

            /* Load H[i][j-1 .. j+14] → left */
            for (int k = 0; k < VL; k++)
                leftBuf[k] = H[i][j-1+k];
            for (int k = VL; k < 16; k++)
                leftBuf[k] = 0;
            __m256i vLeftBase = _mm256_load_si256((__m256i*)leftBuf);

            /* DP transitions */
            __m256i vDiag = _mm256_adds_epi16(vDiagBase, vMat);
            __m256i vUp   = _mm256_adds_epi16(vUpBase,   vGap);
            __m256i vLf   = _mm256_adds_epi16(vLeftBase, vGap);

            __m256i vMax1 = _mm256_max_epi16(vDiag, vUp);
            __m256i vMax2 = _mm256_max_epi16(vLf,  vMax1);
            __m256i vRes  = _mm256_max_epi16(vZero, vMax2);

            /* Write back H[i][j..j+15] */
            int16_t tmp[16];
            _mm256_store_si256((__m256i*)tmp, vRes);

            for (int k = 0; k < VL; k++) {
                H[i][j+k] = tmp[k];
                if (tmp[k] > maxv) maxv = tmp[k];
            }
        }
    }

    /* free */
    for (int i = 0; i <= n; i++) free(H[i]);
    free(H);

    return maxv;
}

#endif

    /* ============================================================
     *            AVX-512 VERSION (32 lanes of int16)
     * ============================================================ */
#ifdef __AVX512BW__

    static int sw_avx512_full(const char *a, const char *b, int n, int m)
{

    /* Allocate full DP matrix H[(n+1) × (m+1)] aligned to 64 bytes */
    int16_t **H = (int16_t**)malloc((n+1) * sizeof(int16_t*));
    for (int i = 0; i <= n; i++) {
        void *p = NULL;
        posix_memalign(&p, 64, (m+1) * sizeof(int16_t));
        H[i] = (int16_t*)p;
        memset(H[i], 0, (m+1) * sizeof(int16_t));
    }

    __m512i vGap  = _mm512_set1_epi16(GAP);
    __m512i vZero = _mm512_set1_epi16(0);

    int maxv = 0;

    /* MAIN DP LOOP */
    for (int i = 1; i <= n; i++) {

        for (int j = 1; j <= m; j += 32) {

            int rem = m - j + 1;
            int VL  = rem >= 32 ? 32 : rem;

            /* --- Build MATCH / MISMATCH vector safely --- */
            int16_t A[32];
            for (int k = 0; k < VL; k++)
                A[k] = (a[i-1] == b[j-1+k]) ? MATCH : MISMATCH;
            for (int k = VL; k < 32; k++) A[k] = 0;
            __m512i vMat = _mm512_load_si512(A);

            /* --- Load H[i-1][j-1 .. j+30] → diag --- */
            int16_t diagBuf[32];
            for (int k = 0; k < VL; k++)
                diagBuf[k] = H[i-1][j-1+k];
            for (int k = VL; k < 32; k++) diagBuf[k] = 0;
            __m512i vDiagBase = _mm512_load_si512(diagBuf);

            /* --- Load H[i-1][j .. j+31] → up --- */
            int16_t upBuf[32];
            for (int k = 0; k < VL; k++)
                upBuf[k] = H[i-1][j+k];
            for (int k = VL; k < 32; k++) upBuf[k] = 0;
            __m512i vUpBase = _mm512_load_si512(upBuf);

            /* --- Load H[i][j-1 .. j+30] → left --- */
            int16_t leftBuf[32];
            for (int k = 0; k < VL; k++)
                leftBuf[k] = H[i][j-1+k];
            for (int k = VL; k < 32; k++) leftBuf[k] = 0;
            __m512i vLeftBase = _mm512_load_si512(leftBuf);

            /* --- Compute DP transitions --- */

            __m512i vDiag = _mm512_adds_epi16(vDiagBase, vMat);
            __m512i vUp   = _mm512_adds_epi16(vUpBase,   vGap);
            __m512i vLeft = _mm512_adds_epi16(vLeftBase, vGap);

            __m512i vMax1 = _mm512_max_epi16(vDiag, vUp);
            __m512i vMax2 = _mm512_max_epi16(vLeft, vMax1);
            __m512i vRes  = _mm512_max_epi16(vZero, vMax2);

            /* --- Store back H[i][j..j+31] safely --- */
            int16_t tmp[32];
            _mm512_store_si512(tmp, vRes);

            for (int k = 0; k < VL; k++) {
                H[i][j+k] = tmp[k];
                if (tmp[k] > maxv) maxv = tmp[k];
            }
        }
    }

    /* Free matrix */
    for (int i = 0; i <= n; i++) free(H[i]);
    free(H);

    return maxv;
}

#endif

void generate_sequence(char *seq, int n) {
    const char alphabet[] = "ACGT";
    for (int i = 0; i < n; i++)
        seq[i] = alphabet[rand() % 4];
    seq[n] = '\0';
}

static inline int sw_cell(char a, char b, int diag, int up, int left) {
    int match = diag + (a == b ? MATCH : MISMATCH);
    int del = up + GAP;
    int ins = left + GAP;
    return MAX(0, MAX(match, MAX(del, ins)));
}



int smith_waterman_optimized(const char *seq1, const char *seq2, int len1, int len2) {
    int **H = malloc((len1 + 1) * sizeof(int *));
    for (int i = 0; i <= len1; i++)
        H[i] = calloc(len2 + 1, sizeof(int));

    int max_score = 0;
    //TODO
    #ifdef __AVX512BW__
        return sw_avx512_full(seq1, seq2, len1, len2);
    #elif defined(__AVX2__)
        return sw_avx2_full(seq1, seq2, len1, len2);
    #elif defined(__SSE4_1__)
        return sw_sse41_full(seq1, seq2, len1, len2);
    #else
        return sw_scalar(seq1, seq2, len1, len2);
    #endif

    for (int i = 0; i <= len1; i++) free(H[i]);
    free(H);

    return max_score;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <sequence_length>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    srand(42);

    char *seq1 = malloc((N + 1) * sizeof(char));
    char *seq2 = malloc((N + 1) * sizeof(char));

    generate_sequence(seq1, N);
    generate_sequence(seq2, N);

    clock_t start = clock();
    int score = smith_waterman_optimized(seq1, seq2, N, N);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Sequence length: %d\n", N);
    printf("Smith-Waterman optimized score: %d\n", score);
    printf("Execution time: %.6f seconds\n", elapsed);

    free(seq1);
    free(seq2);
    return 0;
}
