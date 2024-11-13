//%%cu
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include <stdio.h>
#include <math.h>
#include "../scope_timer.hpp"

#define checkCudaErr(val) check( (val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}


// a[][] * b[][] = c[][]
// 
//                         b00 b01 b02 b03
//                         b10 b11 b12 b13
//                         b20 b21 b22 b23
//                         b30 b31 b32 b33
//
// a00 a01 a02 a03         c00 c01 c02 c03
// a10 a11 a12 a13         c10 c11 c12 c13     block(1, 0) -> shared memory
// a20 a21 a22 a23         c20 c21 c22 c23     c20 c21
// a30 a31 a32 a33         c30 c31 c32 c33     c30 c31
//
//                              b00 b01->  sub_b_step_0
//                              b10 b11
//
//                              b20 b21->  sub_b_step_1
//                              b30 b31
// sub_a_step_0 sub_a_step_1    sub_c
// a20 a21      a22 a23         c20 c21
// a30 a31      a32 a33         c30 c31
//
// sub_c = sub_a_step_0 * sub_b_step_0 + sub_a_step_1 * sub_b_step_1;
//
// for(int step =0; step < N/block_size; step++ )
//      load sub_a_step to shared memory;
//      load sub_b_step to shared memory;
//      tmp += sub_a_step_on_sharedmemory * sub_b_step_on_sharedmemory;
// sub_c = tmp;
//
// cudaMalloc -> global memory
// data:     global memory -> shared memory
// threads:  shared memory -> register
// shared memory SM(stream multi-processor) same block same shared memory
//
// c21 = a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31
// a00 a01 a02 a03 a10 a11 a12 a13 a20 a21 a22 a23 a30 a31 a32 a33
// 0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
// b00 b01 b02 b03 b10 b11 b12 b13 b20 b21 b22 b23 b30 b31 b32 b33

// index = y * size + x
// step 0:3
// a_index = y * size + step
// b_index = step * size + x

#define M 640
#define N 640
#define K 640

__managed__ int a[M*N];
__managed__ int b[N*K];
__managed__ int c_gpu[M*K];
__managed__ int c_cpu[M*K];

__managed__ int c_gpu_2[M*K];
__managed__ int c_gpu_3[M*K];

#define BLOCK_SIZE 16
//#define BLOCK_SIZE 32 // 16 is better than


__global__ void gpu_matrix(int* a, int* b, int* c, int m, int n, int k)
{
    __shared__ int sub_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int sub_b[BLOCK_SIZE][BLOCK_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tmp =0;
    int idx;
    for(int step=0; step <= n/BLOCK_SIZE; step++)
    {
        int step_x = step * BLOCK_SIZE + threadIdx.x;
        int step_y = y;
        idx = step_y * n + step_x;
        if(step_x >= n || step_y >= m)
        {
            sub_a[threadIdx.y][threadIdx.x] =0;
        }
        else
        {
            sub_a[threadIdx.y][threadIdx.x] = a[idx];
        }

        step_x = x;
        step_y = step * BLOCK_SIZE + threadIdx.y;
        idx = step_y * k +step_x;
        if(step_x >= k || step_y >= n)
        {
            sub_b[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            sub_b[threadIdx.y][threadIdx.x] = b[idx];
        }

        __syncthreads();

        for(int i = 0; i < BLOCK_SIZE; i++)
        {
            tmp +=sub_a[threadIdx.y][i] * sub_b[i][threadIdx.x];
        }
        __syncthreads();
    }

    //a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31

    if ( x < k && y < m)
    {
        c[y*k + x] = tmp; 
    }
}

void cpu_matrix(int* a, int* b, int* c, int m, int n, int k)
{
    for( int y = 0; y < m; y++)
    {
        for(int x = 0; x < k; x++)
        {
            int tmp = 0;
            for(int step =0; step < n; step++)
            {
                tmp += a[y*n + step] * b[step*k + x];
            }
            c[y * k + x] = tmp;
        }
    }
}

__global__ void multiplicateMatrixOnDevice(int *array_A, int *array_B, int *array_C, int M_p, int K_p, int N_p)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;

    if (ix < N_p && iy < M_p) 
    {
        int sum = 0;
        for (int k = 0; k < K_p; k++)
        {
            sum += array_A[iy*K_p + k] * array_B[k*N_p + ix];
        }
        array_C[iy*N_p + ix] = sum;
    }

}

/*
* 设定每一个thread负责C中的一个坐标的矩阵乘，一个需要numCRows x numCColumns 线程
* 使用share memory的好处是尽量减少访问global memory的次数，从而降低计算时间。
* 把一小块数据称为一个sub block，加入sharememory之后，就可以被C_sub_block计算复用。
* 如上层 一个线程在计算c21时，用到了 a20 a21 a22 a23
* 另外一个线程在计算c22时， 也要使用 a20 a21 a22 a23
* 那这个时候把 a20 a21 a22 a23 放在share memory中，就可以降低global memory的访存次数。
* 在同一个block内，是共享share memory的。
*/
__global__ void matrixMultiplyShared(int *A, int *B, int *C, 
    int numARows, int numAColumns, 
    int numBRows, int numBColumns, 
    int numCRows, int numCColumns)
{
    __shared__ int sharedM[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int sharedN[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    int Csub = 0;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (int)(ceil((float)numAColumns/BLOCK_SIZE)); i++)
    {
        //Each thread loads one element of each sub-matrix
        //Load Asub and Bsub from device memory to shared memory
        if (i * BLOCK_SIZE+tx < numAColumns && row < numARows)
            sharedM[ty][tx] = A[row*numAColumns + i * BLOCK_SIZE + tx];
        else
            sharedM[ty][tx] = 0;

        if (i * BLOCK_SIZE+ty < numBRows && col < numBColumns)
            sharedN[ty][tx] = B[(i*BLOCK_SIZE + ty)*numBColumns + col];
        else
            sharedN[ty][tx] = 0;
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        //c21 = a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31
        // one thread dedicated to calcuate c21
        for (int j = 0; j < BLOCK_SIZE; j++)
            Csub += sharedM[ty][j] * sharedN[j][tx];
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();

    }

    //Write Csub to device memory
    //Each thread writes one element
    if (row < numCRows && col < numCColumns)
        C[row*numCColumns + col] = Csub;

        

}

int main()
{
    for(int y=0; y<M; ++y)
    {
        for(int x=0; x<N; ++x)
        {
            a[y * N + x] = rand()%1024;
        }
    }

    for(int y=0; y<N; ++y)
    {
        for(int x=0; x<K; ++x)
        {
            b[y*K + x] = rand()%1024;
        }
    }

    unsigned int grid_x = (K + BLOCK_SIZE -1)/BLOCK_SIZE;
    unsigned int grid_y = (M + BLOCK_SIZE -1)/BLOCK_SIZE;

    dim3 dimGrid(grid_x, grid_y);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); // 1. gpu warp 32,   2.  <=1024
    {
        ScopeTimer<TimerPlatForm::GPU> timer("GPU Matrix Multiply: ");
        gpu_matrix<<<dimGrid, dimBlock>>>(a, b, c_gpu, M, N, K);
    }

    {
        ScopeTimer<TimerPlatForm::GPU> timer("GPU Matrix Multiply: ");
        gpu_matrix<<<dimGrid, dimBlock>>>(a, b, c_gpu, M, N, K);
    }
    
    {
        ScopeTimer<TimerPlatForm::CPU> timer("CPU Matrix Multiply: ");
        cpu_matrix(a, b, c_cpu, M, N, K);
    }

    {
        ScopeTimer<TimerPlatForm::GPU> timer("GPU Matrix Multiply without sharedmem: ");
        multiplicateMatrixOnDevice<<<dimGrid, dimBlock>>>(a, b, c_gpu_2, M, N, K);
    }
    
    {
        ScopeTimer<TimerPlatForm::GPU> timer("GPU Matrix Multiply sm v2: ");
        matrixMultiplyShared<<<dimGrid, dimBlock>>>(a, b, c_gpu_3, M, N, N,K,M,K);
    }
    


    

    bool errors = false;

    for(int y=0; y<M; y++)
    {
        for(int x=0; x<K; x++)
        {
            if(fabs(c_cpu[y*K + x] - c_gpu[y*K+x]) > (1.0e-10))
            {
                errors = true;
                printf("c_cpu: %d. c_gpu: %d", c_cpu[y*K + x], c_gpu[y*K+x]);
            }
        }
    }

    printf("Result: %s\n", errors?"Error":"Pass");

    printf("test2 test =====>\n");
    for(int y=0; y<M; y++)
    {
        for(int x=0; x<K; x++)
        {
            if(fabs(c_cpu[y*K + x] - c_gpu_2[y*K+x]) > (1.0e-10))
            {
                errors = true;
                printf("c_cpu: %d. c_gpu_2: %d diff = %d\n", c_cpu[y*K + x], c_gpu_2[y*K+x], c_cpu[y*K + x] - c_gpu_2[y*K+x]);
                break;
            }
        }
        if (errors)
            break;
    }
    printf("Result: %s\n", errors?"Error":"Pass");

    printf("test3 test =====>\n");
    for(int y=0; y<M; y++)
    {
        for(int x=0; x<K; x++)
        {
            if(fabs(c_cpu[y*K + x] - c_gpu_3[y*K+x]) > (1.0e-10))
            {
                errors = true;
                printf("c_cpu: %d. c_gpu_3: %d diff = %d\n", c_cpu[y*K + x], c_gpu_3[y*K+x], c_cpu[y*K + x] - c_gpu_3[y*K+x]);
                break;
            }
        }
        if (errors)
            break;
    }
    printf("Result: %s\n", errors?"Error":"Pass");

    return 0;
}
