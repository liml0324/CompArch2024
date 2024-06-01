#include<bits/stdc++.h>

using namespace std;

int N = (1 << 12);

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
    clock_t start = clock();
    gemm_baseline(A, B, C);
    clock_t end = clock();
    cout << "Time: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;
    // free A, B, C
    free(A);
    free(B);
    free(C);
    return 0;
}
void gemm_baseline(float *A, float *B, float *C) {
    memset(C, 0, N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}
