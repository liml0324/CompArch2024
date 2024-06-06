#include<bits/stdc++.h>
#include<omp.h>

using namespace std;

#ifndef N
#define N (1 << 12)
#endif

void gemm_baseline(float *A, float *B, float *C); // you can use inline function

int main(void) {
    // malloc A, B, C
    float *A = (float *)malloc(N * N * sizeof(float));
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C = (float *)malloc(N * N * sizeof(float));
    // random initialize A, B
    default_random_engine generator((unsigned)time(NULL));
    uniform_real_distribution<float> distribution(-10, 10);
    for (int i = 0; i < N * N; i++) {
        A[i] = distribution(generator);
        B[i] = distribution(generator);
    }
    // measure time
    auto start = omp_get_wtime();
    gemm_baseline(A, B, C);
    auto end = omp_get_wtime();
    cout << "Time: " << 1000 * (end - start) << "ms" << endl;
    // free A, B, C
    free(A);
    free(B);
    free(C);
    return 0;
}
void gemm_baseline(float *A, float *B, float *C) {
    memset(C, 0, N * N * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}
