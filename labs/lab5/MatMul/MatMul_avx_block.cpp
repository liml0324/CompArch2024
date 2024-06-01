#include <bits/stdc++.h>
#include <immintrin.h>
// #define VERIFY

using namespace std;

int N = (1 << 12);

void gemm_verify(float *A, float *B, float *C); // you can use inline function

// you may need to add some additional function parameters to adjust the blocking  strategy.
void gemm_avx_block(float *A, float *B, float *C, int blockSize); // you can use inline function
float _mm256_reduce_add_ps(__m256 a);

int main(void) {
    // malloc A, B, C
    float *A = (float *)malloc(N * N * sizeof(float));
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C = (float *)malloc(N * N * sizeof(float));
    #ifdef VERIFY
    float *D = (float *)malloc(N * N * sizeof(float));
    #endif
    // random initialize A, B
    default_random_engine generator((unsigned)time(NULL));
    uniform_real_distribution<float> distribution(-1, 1);
    for (int i = 0; i < N * N; i++) {
        A[i] = distribution(generator);
        B[i] = distribution(generator);
    }
    // measure time
    clock_t start = clock();
    gemm_avx_block(A, B, C, 16);
    clock_t end = clock();
    cout << "Time: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;
    // use gemm_baseline verify gemm_avx_block
    #ifdef VERIFY
    gemm_verify(A, B, D);
    for (int i = 0; i < N * N; i++) {
        if (abs(C[i] - D[i]) > 1e-3) {
            cout << "Error: " << i << " " << C[i] << " " << D[i] << endl;
            break;
        }
    }
    #endif
    // free A, B, C, D
    free(A);
    free(B);
    free(C);
    #ifdef VERIFY
    free(D);
    #endif
    return 0;
}
void gemm_verify(float *A, float *B, float *C) {
    memset(C, 0, N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}
float _mm256_reduce_add_ps(__m256 a) {
    return a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7];
}
void gemm_avx_block(float *A, float *B, float *C, int blockSize) {
    float *B_T = (float *)malloc(N * N * sizeof(float));
    // transpose B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B_T[i * N + j] = B[j * N + i];
        }
    }
    __m256 sum, a, b;
    for (int i = 0; i < N; i += blockSize) {
        for (int j = 0; j < N; j += blockSize) {
            for (int k = 0; k < N; k += blockSize) {
                for (int ii = i; ii < i + blockSize; ++ii) {
                    for (int jj = j; jj < j + blockSize; ++jj) {
                        sum = _mm256_setzero_ps();
                        for (int kk = k; kk < k + blockSize; kk += 8) {
                            a = _mm256_loadu_ps(A + ii * N + kk);
                            b = _mm256_loadu_ps(B_T + jj * N + kk);
                            sum = _mm256_fmadd_ps(a, b, sum);
                        }
                        C[ii * N + jj] += _mm256_reduce_add_ps(sum);
                    }
                }
            }
        }
    }
}
