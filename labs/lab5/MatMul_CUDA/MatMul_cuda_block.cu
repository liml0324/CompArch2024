#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define BLOCK_LEN 32
// #define VERIFY

#define N (1 << 15)

__global__ void gemm_block(float *A, float *B, float *C, int n);
void gemm_verify(float *A, float *B, float *C);

using namespace std;

int main() {
    // malloc A, B, C
    float *A = (float *)malloc(N * N * sizeof(float));
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C = (float *)malloc(N * N * sizeof(float));
    // random initialize A, B
    default_random_engine generator((unsigned)time(NULL));
    uniform_real_distribution<float> distribution(-1, 1);
    for (int i = 0; i < N * N; i++) {
        A[i] = distribution(generator);
        B[i] = distribution(generator);
    }
    // cumalloc A, B, C
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));
    // define gridsize and blocksize
    dim3 gridsize(N/BLOCK_LEN, N/BLOCK_LEN);
    dim3 blocksize(BLOCK_LEN, BLOCK_LEN);
    // copy A, B to device
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    // launch kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    gemm_block<<<gridsize, blocksize>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float *D = (float *)malloc(N * N * sizeof(float));
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time: " << milliseconds << "ms" << endl;
    cudaMemcpy(D, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    // compute
    #ifdef VERIFY
    gemm_verify(A, B, C);
    for (int i = 0; i < N * N; i++) {
        if (abs(C[i] - D[i]) > 1e-3) {
            cout << "Error: " << i << " " << C[i] << " " << D[i] << endl;
            break;
        }
    }
    free(D);
    #endif
    // free mem
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

__global__ void gemm_block(float *A, float *B, float *C, int n) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_LEN + ty;
    int col = bx * BLOCK_LEN + tx;

    __shared__ float As[BLOCK_LEN][BLOCK_LEN];
    __shared__ float Bs[BLOCK_LEN][BLOCK_LEN];

    float Csub = 0;

    for (int t = 0; t < n / BLOCK_LEN; t++) {
        As[ty][tx] = A[row * n + t * BLOCK_LEN + tx];
        Bs[ty][tx] = B[(t * BLOCK_LEN + ty) * n + col];
        __syncthreads();

        for (int i = 0; i < BLOCK_LEN; i++) {
            Csub += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }

    C[row * n + col] = Csub;

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