#include<bits/stdc++.h>
#include<immintrin.h>
// #define VERIFY

using namespace std;

int N = (1 << 12);

void gemm_verify(float *A, float *B, float *C); // you can use inline function
void gemm_avx(float *A, float *B, float *C); // you can use inline function

int main(void) {
    // malloc A, B, C
    float *A = (float *)malloc(N * N * sizeof(float));
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C = (float *)malloc(N * N * sizeof(float));
    float *D = (float *)malloc(N * N * sizeof(float));
    // random initialize A, B
    default_random_engine generator((unsigned)time(NULL));
    uniform_real_distribution<float> distribution(-1, 1);
    for (int i = 0; i < N * N; i++) {
        A[i] = distribution(generator);
        B[i] = distribution(generator);
    }
    // measure time
    clock_t start = clock();
    gemm_avx(A, B, C);
    clock_t end = clock();
    cout << "Time: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;
    // verify
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
    free(D);
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
void gemm_avx(float *A, float *B, float *C) {
    float *B_T = (float *)malloc(N * N * sizeof(float));
    // transpose B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B_T[i * N + j] = B[j * N + i];
        }
    }
    __m256 sum, a, b;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            sum = _mm256_setzero_ps();
            for (int k = 0; k < N; k+=8) {
                a = _mm256_loadu_ps(&A[i * N + k]);
                b = _mm256_loadu_ps(&B_T[j * N + k]);
                sum = _mm256_fmadd_ps(a, b, sum);
            }
            C[i * N + j] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
        }
    }
    free(B_T);
}
