#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N (1 << 10)

__global__ void gemm_baseline(float *A, float *B, float *C);
void gemm_verify(float *A, float *B, float *C);

using namespace std;

int main()
{
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
    // cumalloc A, B, C
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));
    // define gridsize and blocksize
    dim3 gridsize(1, 1, 1);
    dim3 blocksize(N, N, 1);
    // copy A, B to device
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    // launch kernel
    gemm_baseline<<<gridsize, blocksize>>>(d_A, d_B, d_C);
    // compute
        // gemm_verify(A, B, C);
    // free mem
}
__global__ void gemm_baseline(float* A, float * B, float* C) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    float sum = 0;
    for (int k = 0; k < N; k++) {
        sum += A[i * N + k] * B[k * N + j];
    }
    C[i * N + j] = sum;
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
