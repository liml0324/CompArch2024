# Lab 5 Report
#### PB21111639 李牧龙
## 代码实现
### 基础矩阵乘法
很简单，直接三层循环解决。
```c
memset(C, 0, N * N * sizeof(float));
for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] += A[i * N + k] * B[k * N + j];
        }
    }
}
```
注意这里先对k循环，再对j循环。这样让对B的访问变为按行访问，有利于缓存命中。

### AVX矩阵乘法
#### 基础AVX矩阵乘法
AVX指令集可以计算长度为8的float向量的乘法，而传统矩阵乘法（按ijk顺序进行循环，而非上面的ikj顺序）最内层可以看作向量点乘。因此可以将传统矩阵乘法最内层的N个乘法改为用AVX指令集进行计算，即可将最内层由N次float乘法变为N/8次AVX乘法。
由于AVX的`_mm256_loadu_ps`方法只能将连续的数据load进寄存器，因此需要对矩阵B进行转置，由按行存储变为按列存储。
```c
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
            a = _mm256_loadu_ps(&A[i * N + k]);     // load 8 floats from A
            b = _mm256_loadu_ps(&B_T[j * N + k]);   // load 8 floats from B_T
            sum = _mm256_fmadd_ps(a, b, sum);       // sum += a * b
        }
        C[i * N + j] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
    }
}
free(B_T);
```

#### AVX分块乘法
使用类似PPT中的方法计算。每次计算C中的一个块，这个块由A和B中的多个块进行乘法再求和得到。
同样先对B进行转置。
```c
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
                    // 这里对sum中8个float求和，加到C中
                    // _mm256_reduce_add_ps是自己写的函数
                    C[ii * N + jj] += _mm256_reduce_add_ps(sum);
                }
            }
        }
    }
}
free(B_T);
```

### CUDA矩阵乘法
#### CUDA基础矩阵乘法
基础CUDA矩阵乘法的实现很简单，让每个CUDA线程计算C中的一个元素，即每个CUDA线程执行一个长度为N的向量点乘。
由于CUDA中每个线程块中最多只能有1024个线程，因此在矩阵元素多于1024个时需要用多个线程块进行计算。

```c
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
}
```
调用时这样调用：
```c
dim3 gridsize(N/BLOCK_LEN, N/BLOCK_LEN);
dim3 blocksize(BLOCK_LEN, BLOCK_LEN);
gemm_baseline<<<gridsize, blocksize>>>(d_A, d_B, d_C, N);
```
其中`BLOCK_LEN`用来限制每个线程块的大小。一般取32。

#### CUDA分块矩阵乘法
原理与AVX分块乘法类似，一个线程块负责计算C中的一个子矩阵，这个子矩阵由A和B中的多个子矩阵进行乘法再求和得到。在每次进行子矩阵的乘法前，先将矩阵元素load进共享内存中，一个线程块中的每个线程负责载入一个元素。这样对于每个线程而言，只需增加一次载入的代价，就可极大地减少接下来的若干次浮点数乘法（实际是一个向量点乘）的访存代价。

```c
__global__ void gemm_block(float *A, float *B, float *C, int n) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_LEN + ty;  // load矩阵A中元素时的行号
    int col = bx * BLOCK_LEN + tx;  // load矩阵B中元素时的列号

    __shared__ float As[BLOCK_LEN][BLOCK_LEN];
    __shared__ float Bs[BLOCK_LEN][BLOCK_LEN];

    float Csub = 0;

    for (int t = 0; t < n / BLOCK_LEN; t++) {
        As[ty][tx] = A[row * n + t * BLOCK_LEN + tx];   // 每个线程load一个元素
        Bs[ty][tx] = B[(t * BLOCK_LEN + ty) * n + col];
        __syncthreads();

        for (int i = 0; i < BLOCK_LEN; i++) {   // 利用线程块内共享的As和Bs计算Csub
            Csub += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }

    C[row * n + col] = Csub;

}
```

## 编译运行
### 编译
我在src目录下提供了Makefile，可以利用此Makefile来进行编译。

- 使用`make cpp`即可编译所有的CPU矩阵乘法代码，编译后的可执行文件位于`build`目录下。
- 使用`make cuda`即可编译所有的CUDA矩阵乘法代码，编译后的可执行文件位于`build`目录下。
- 使用`make all`可以编译所有代码。
- 使用`make clean`可以清除所有编译生成的文件。

另外，可以通过更改Makefile中的`MACRO`来修改矩阵大小、分块大小等参数。其中`-DN=(1 << 12)`指定了矩阵大小，`-DBLOCK_LEN=16`指定了分块大小（对于CUDA矩阵乘法，这一参数不能大于32），`-DVERIFY`如果存在，则会对计算结果进行验证。

### 运行
直接运行编译好的`.o`文件即可。

## 测试结果
### 测试环境和数据
#### 测试平台
- CPU: 10-Core 12th Gen Intel Core i7-12700KF (-MT MCP-) 3610 MHz
- GPU: NVIDIA GeForce RTX 4070 SUPER
- Mem: DDR5 5986MHz 15906.5 MiB
- Kernel: 5.15.146.1-microsoft-standard-WSL2 x86_64
- CUDA: V12.4.131

注：
- CPU和内存实际分别为20线程，32G，linux中只能查询到一半，可能是系统只给WSL2分配了一半资源。
- 操作系统是使用Ubuntu 20.04的WSL2，基于Windows 11。
#### 测试数据
测试数据为使用`std::default_random_engine`随机生成的在[-1, 1]之间的浮点数，以时间为种子。
### 正确性测试
以基础矩阵乘法（即简单的3层循环）作为正确结果，若所有计算结果与正确结果的差距均小于1e-3，则认为计算正确。
由于基础矩阵乘法的计算速度较慢，因此仅对256\*256和1024\*1024规模的各类矩阵乘法进行了正确性测试，结果均正确。

### 性能测试
以下所有测试结果均使用`-O2`编译优化。
新版CUDA设备不支持`nvprof`工具，改为使用Nsight系列工具。但该系列工具生成的测试结果查看起来不太方便，因此实际测试中使用`cudaEvent_t`进行时间测量，对于普通的计时需求，这种方法已经足够。
#### 基础矩阵乘法
结果如下：
| N | Time(ms) |
|---|----------|
| 256 | 5.396 |
| 512 | 37.617 |
| 1024 | 305.745 |
| 2048 | 2533.27 |
| 4096 | 23559.6 |

#### AVX矩阵乘法
| N | Time(ms) |
|---|----------|
| 256 | 1.807 |
| 512 | 12.757 |
| 1024 | 113.517 |
| 2048 | 1009.62 |
| 4096 | 11687.5 |

#### AVX分块矩阵乘法
取分块大小为64*64
| N | Time(ms) |
|---|----------|
| 256 | 1.876 |
| 512 | 11.281 |
| 1024 | 85.113 |
| 2048 | 695.741 |
| 4096 | 5820.21 |

从上面的测试结果可以看出，AVX矩阵乘法相比基础矩阵乘法有了明显的提升，而AVX分块矩阵乘法相比AVX矩阵乘法有了更大的提升。

#### 分块大小对性能的影响
取N为4096
| 分块大小 | Time(ms) |
|----------|----------|
| 8*8 | 14707.5 |
| 16*16 | 8874.91 |
| 32*32 | 6522.06 |
| 64*64 | 5820.21 |
| 128*128 | 8447.23 |
| 256*256 | 8150.62 |
| 512*512 | 7808.75 |
| 1024*1024 | 9133.71 |

可以看出，64*64时性能最好。分块太小或太大都会导致性能下降，甚至比基础矩阵乘法还慢。

#### CUDA矩阵乘法
线程块的大小设置为32*32
| N | Time(ms) |
|---|----------|
| 1024 | 4.94906 |
| 2048 | 9.92301 |
| 4096 | 72.868 |
| 8192 | 536.005 |
| 16384 | 4098.8 |
| 32768 | 33298.3 |

#### CUDA分块矩阵乘法
分块大小设置为32*32
| N | Time(ms) |
|---|----------|
| 1024 | 3.35968 |
| 2048 | 8.224 |
| 4096 | 56.6295 |
| 8192 | 417.332 |
| 16384 | 3207.63 |
| 32768 | 26480.3 |

可以看出，CUDA矩阵乘法远远快于各种CPU矩阵乘法，而CUDA分块矩阵乘法相比CUDA基础矩阵乘法更快，但提升不如CPU的分块矩阵乘法明显。

#### 线程块大小对基础CUDA矩阵乘法性能的影响
N为16384
| 线程块大小 | Time(ms) |
|----------|----------|
| 8*8 | 18279.2 |
| 16*16 | 5146.03 |
| 32*32 | 4098.8 |
可以看到，线程块越大性能越好，因为更大的线程块可以更好地利用硬件的计算资源。

#### 分块大小对CUDA分块矩阵乘法性能的影响
N为16384
| 分块大小 | Time(ms) |
|----------|----------|
| 8*8 | 7861.75 |
| 16*16 | 4320 |
| 32*32 | 3207.63 |

同样是分块越大性能越好，分块越大除了能更好地利用硬件资源外，还能更好地利用共享内存，减少访存次数。

## CPU矩阵乘法的其他优化方法
- 多线程优化：使用OpenMP等多线程库，将矩阵乘法的三层循环分配给多个线程，提高CPU的利用率。我在基础矩阵乘法的基础上，通过简单地添加`#pragma omp parallel for`，将N=4096的矩阵乘法的时间从23559.6ms降低到了2538.26ms（开启O2优化），甚至显著快于AVX分块矩阵乘法。源代码位于`MatMul_omp.cpp`中，需要通过`g++ -O2 -fopenmp MatMul_omp.cpp`编译。
- 循环交换：通过改变多层循环的顺序来提高访存的连续性。本次实验中CPU矩阵乘法的baseline就已使用了这种方法。
- 调用高性能库函数：BLAS等库中的矩阵乘法函数通常经过高度优化，可以直接调用这些函数来提高性能。

## 总结
本次实验中，我实现了基础矩阵乘法、AVX矩阵乘法、AVX分块矩阵乘法、CUDA矩阵乘法和CUDA分块矩阵乘法，并对这些方法进行了性能测试。从测试结果来看，CUDA矩阵乘法的性能远远高于CPU矩阵乘法，而CUDA分块矩阵乘法相比CUDA基础矩阵乘法有了一定的提升，但提升不如CPU的分块矩阵乘法明显。在CPU矩阵乘法中，除向量优化外，使用OpenMP等多线程库可以显著提高性能。在实际应用中，可以根据具体情况选择合适的矩阵乘法方法。