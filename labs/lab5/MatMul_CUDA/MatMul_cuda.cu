#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// #define VERIFY

#define N (1 << 14)

__global__ void gemm_baseline(float *A, float *B, float *C, int n);
void gemm_verify(float *A, float *B, float *C);

using namespace std;

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int max_block_thread_num = prop.maxThreadsPerBlock;
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
    double temp = (double)N / sqrt(max_block_thread_num);
    int block_num = ceil(temp);
    int i = 1;
    while(i > 0) {
        if(i < block_num)   i <<= 1;
        else    break;
    }
    block_num = i;
    dim3 gridsize(block_num, block_num);
    dim3 blocksize(N/block_num, N/block_num);
    // copy A, B to device
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    // launch kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    gemm_baseline<<<gridsize, blocksize>>>(d_A, d_B, d_C, N);
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
__global__ void gemm_baseline(float* A, float * B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= n || col >= n) {
        return;
    }
    float sum = 0;
    for (int k = 0; k < n; k++) {
        sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
    // printf("row: %d, col: %d, sum: %f\n", row, col, sum);
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
