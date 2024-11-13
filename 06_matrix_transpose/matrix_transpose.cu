/**
 * reference:
 * https://blog.csdn.net/kunhe0512/article/details/131405148?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171707319216800225560928%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=171707319216800225560928&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-2-131405148-null-null.nonecase&utm_term=%E7%9F%A9%E9%98%B5%E8%BD%AC%E7%BD%AE&spm=1018.2226.3001.4450
 * 
*/

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


#define M 16
#define N 16


__managed__ int a[M*N];

__managed__ int c_cpu[N*M];

__managed__ int c_gpu[N*M];
__managed__ int c_gpu_2[N*M];
__managed__ int c_gpu_3[N*M];

#define BLOCK_SIZE 32

#define TILE_DIM 16

/**
 * 合并访存，让连续的线程访问连续的内存
*/

void print_matrix(int *array, int size_y, int size_x)
{
    printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    
    for(int y=0; y<size_y; y++)
    {
        for(int x=0; x<size_x; x++)
        {
            printf("%4d ", array[y*size_x + x]);
        
        }
        printf("\n");
    }
    printf("---------------------------------------------------------\n");
}

void cpu_matrix_transpose(int* a, int* c, int m, int n)
{
    for( int y = 0; y < n; y++)
    {
        for(int x = 0; x < m; x++)
        {
            c[y * m + x] = a[x*n + y];
        }
    }
}

void cpu_matrix_transpose_1(int* a, int* c, int m, int n)
{
    int *pos = c;
    for( int y = 0; y < n; y++)
    {
        for(int x = 0; x < m; x++)
        {
            *pos++ = a[x*n + y];
        }
    }
}

void cpu_matrix_transpose_2(int* a, int* c, int m, int n)
{
    int *pos = a;
    for( int y = 0; y < m; y++)
    {
        for(int x = 0; x < n; x++)
        {
            c[x * m + y] = *pos++;
            //c[x * m + y] = a[y*n +x];
            //printf("(%d, %d) -> (%d %d) %d\n",x, y, y,x , a[y*n+x]);
        }
    }
}


__global__ void gpu_matrix_transpose(int *a, int *c, int m, int n)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < m && y < n) 
    {
        c[y * m + x] = a[x * n + y];
    }
}


__global__ void gpu_matrix_transpose_2(int *a, int *c, int m, int n)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < m && y < n) 
    {
        c[y * M + x] = a[x*N + y];
        //c[x * n + y] = a[y * m + x];
        //printf("a%d%d (%4d %4d) <--> %4d\n", x, y, x, y, a[y * m + x] );
    }
}



__global__ void transpose1(int *A, int *B, const int size)
{
    __shared__ int S[TILE_DIM][TILE_DIM];
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    int nx1 = bx + threadIdx.x;     // 第nx1列
    int ny1 = by + threadIdx.y;     // 第ny1行
    if (nx1 < size && ny1 < size)
    {
        // A的访问是合并的， 第 11 行对共享内存的访问不导致 bank 冲突
        S[threadIdx.y][threadIdx.x] = A[ny1 * size + nx1];
    }
    __syncthreads();
    
    int nx2 = bx + threadIdx.y;     // 第bx块的y行 -》S的x行
    int ny2 = by + threadIdx.x;     // 第by块的x列 -》S的y列
    if (nx2 < size && ny2 < size)
    {
        // 二维的情况：一般32个连续的threadIdx.x构成一个线程束，一个线程束内的threadIdx.y是相同的
        // 同一个线程束内的线程（连续的threadIdx.x）刚好访问同一个bank中的32个数据，这将导致 32 路 bank 冲突
        B[nx2 * size + ny2] = S[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose2(int *A, int *B, const int size)
{
    /*
        这样改变共享内存数组的大小之后，同一个线程束中的 32 个线程（连续的 32 个 threadIdx.x 值）将对应共享内存数组 S 中跨度为 33 的数据。如果第一个线程访问第一个 bank 的第一层，第二个线程则会
        访问第二个 bank 的第二层（而不是第一个 bank 的第二层）；如此等等。于是，这 32 个线程将分别访问 32 个不同 bank 中的数据，所以没有 bank 冲突，
    */
    __shared__ int S[TILE_DIM][TILE_DIM + 1];
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;
    if (nx1 < size && ny1 < size)
    {
        S[threadIdx.y][threadIdx.x] = A[ny1 * size + nx1];
    }
    __syncthreads();

    int nx2 = bx + threadIdx.y;
    int ny2 = by + threadIdx.x;
    if (nx2 < size && ny2 < size)
    {
        B[nx2 * size + ny2] = S[threadIdx.x][threadIdx.y];
    }
}


//              matrix transpose
//                                                     t56 t57 t58
//               in        b00 b01 b02 | b03 b04 b05 | b06 b07 b08  B[6][9]
//                         b10 b11 b12 | b13 b14 b15 | b16 b17 b18
//                         b20 b21 b22 | b23 b24 b25 | b26 b27 b28
//                         ------------+-------------+------------
//                         b30 b31 b32 | b33 b34 b35 | b36 b37 b38
//                         b40 b41 b41 | b43 b44 b45 | b46 b47 b48
//                         b50 b51 b52 | b53 b54 b55 | b56 b57 b58   threadIdx.x=1, threadIdx.y=2;
//                                                                   block 1, 2
//
//                         
//               out       b00 b10 b20 | b30 b40 b50
//                         b01 b11 b21 | b31 b41 b51
//                         b02 b12 b22 | b32 b42 b52
//                         ------------+------------
//                         b03 b13 b23 | b33 b43 b53
//                         b04 b14 b24 | b34 b44 b54
//                         b05 b15 b25 | b35 b45 b55
//                         ------------+------------
//                         b06 b16 b26 | b36 b46 b56
//                         b07 b17 b27 | b37 b47 b57
//                         b08 b18 b28 | b38 b48 b58                 block 2, 1
// shared memory 
// t57 read b57 from global memroy to shared memroy
// t57 read b48 from shared memory
// t57 write b48 to global memory

__global__ void gpu_shared_matrix_transpose(int *a, int *c, int size_x, int size_y)
{
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ int ken[BLOCK_SIZE+1][BLOCK_SIZE+1];//ken[32] warp

    if(x < size_x && y < size_y)
    {
        // read from global memory to shared memory
        ken[threadIdx.y][threadIdx.x] = a[y * size_x + x];
    }
    __syncthreads();

    int x1 = threadIdx.x + blockDim.y * blockIdx.y;
    int y1 = threadIdx.y + blockDim.x * blockIdx.x;
    if(x1 < size_y && y1 < size_x)
    {
        c[y1 * size_y +x1] = ken[threadIdx.x][threadIdx.y];//32 bank
    }
}


__global__ void gpu_shared_matrix_transpose_2(int *a, int *c, int size_x, int size_y)
{
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ int ken[BLOCK_SIZE][BLOCK_SIZE];//ken[32] warp

    if(x < size_x && y < size_y)
    {
        // read from global memory to shared memory
        ken[threadIdx.y][threadIdx.x] = a[y * size_x + x];
    }
    __syncthreads();

    int x1 = threadIdx.x + blockDim.y * blockIdx.y;
    int y1 = threadIdx.y + blockDim.x * blockIdx.x;
    if(x1 < size_y && y1 < size_x)
    {
        c[y1 * size_y +x1] = ken[threadIdx.x][threadIdx.y];//32 bank
    }
}

int main()
{
    int tmp = 0;
    for(int y=0; y<M; ++y)
    {
        for(int x=0; x<N; ++x)
        {
            a[y * N + x] = rand()%1024;
           // a[y * N + x] = tmp++;
        }
    }
    //print_matrix(a, M, N);

    unsigned int grid_x = (M + BLOCK_SIZE -1)/BLOCK_SIZE;
    unsigned int grid_y = (N + BLOCK_SIZE -1)/BLOCK_SIZE;

    dim3 dimGrid(grid_x, grid_y);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); // 1. gpu warp 32,   2.  <=1024
    {
        ScopeTimer<TimerPlatForm::GPU> timer("GPU Matrix Transpose warmup: ");
        gpu_matrix_transpose<<<dimGrid, dimBlock>>>(a, c_gpu, M, N);
    }

    {
        ScopeTimer<TimerPlatForm::GPU> timer("GPU Matrix Transpose: ");
        gpu_matrix_transpose<<<dimGrid, dimBlock>>>(a, c_gpu, M, N);
    }
    //print_matrix(c_gpu, N, M);

    {
        ScopeTimer<TimerPlatForm::GPU> timer("GPU Matrix Transpose 2: ");
        gpu_matrix_transpose_2<<<dimGrid, dimBlock>>>(a, c_gpu_2, M, N);
    }
    //print_matrix(c_gpu_2, N, M);

    {
        ScopeTimer<TimerPlatForm::GPU> timer("GPU Matrix Transpose 2: ");
        gpu_shared_matrix_transpose<<<dimGrid, dimBlock>>>(a, c_gpu_2, N, M);
    }
    //print_matrix(c_gpu_2, N, M);

    {
        ScopeTimer<TimerPlatForm::GPU> timer("GPU Matrix Transpose 2 bf: ");
        gpu_shared_matrix_transpose_2<<<dimGrid, dimBlock>>>(a, c_gpu_2, N, M);
    }
    //print_matrix(c_gpu_2, N, M);

    


    // {
    //     ScopeTimer<TimerPlatForm::GPU> timer("GPU Matrix Transpose share memory w bf: ");
    //     transpose1<<<dimGrid, dimBlock>>>(a, c_gpu_2, M);
    // }
    // //print_matrix(c_gpu_2, N, M);


    // {
    //     ScopeTimer<TimerPlatForm::GPU> timer("GPU Matrix Transpose share memory wo bf: ");
    //     transpose2<<<dimGrid, dimBlock>>>(a, c_gpu_2, M);
    // }
    //print_matrix(c_gpu_2, N, M);

    

    {
        ScopeTimer<TimerPlatForm::CPU> timer("CPU Matrix Transpose: ");
        cpu_matrix_transpose(a, c_cpu, M, N);
    }
    //print_matrix(c_cpu, N, M);
    memset(c_cpu, 0, sizeof(c_cpu));


    {
        ScopeTimer<TimerPlatForm::CPU> timer("CPU Matrix Transpose 1 : ");
        cpu_matrix_transpose_1(a, c_cpu, M, N);
    }
    //print_matrix(c_cpu, N, M);
    memset(c_cpu, 0, sizeof(c_cpu));

    {
        ScopeTimer<TimerPlatForm::CPU> timer("CPU Matrix Transpose 2: ");
        cpu_matrix_transpose_2(a, c_cpu, M, N);
    }
    //print_matrix(c_cpu, N, M);
    //memset(c_cpu, 0, sizeof(c_cpu));

    // {
    //     ScopeTimer<TimerPlatForm::GPU> timer("GPU Matrix Multiply without sharedmem: ");
    //     multiplicateMatrixOnDevice<<<dimGrid, dimBlock>>>(a, b, c_gpu_2, M, N, K);
    // }
    
    // {
    //     ScopeTimer<TimerPlatForm::GPU> timer("GPU Matrix Multiply sm v2: ");
    //     matrixMultiplyShared<<<dimGrid, dimBlock>>>(a, b, c_gpu_3, M, N, N,K,M,K);
    // }
    


    

    bool errors = false;

    for(int y=0; y<N; y++)
    {
        for(int x=0; x<M; x++)
        {
            if(fabs(c_cpu[y*M + x] - c_gpu[y*M+x]) > (1.0e-10))
            {
                errors = true;
                printf("c_cpu: %d. c_gpu: %d", c_cpu[y*M + x], c_gpu_2[y*M+x]);
                break;
            }
        }
        if (errors)
            break;
    }

    printf("\nResult: %s\n", errors?"Error":"Pass");

    // printf("test2 test =====>\n");
    // for(int y=0; y<M; y++)
    // {
    //     for(int x=0; x<K; x++)
    //     {
    //         if(fabs(c_cpu[y*K + x] - c_gpu_2[y*K+x]) > (1.0e-10))
    //         {
    //             errors = true;
    //             printf("c_cpu: %d. c_gpu_2: %d diff = %d\n", c_cpu[y*K + x], c_gpu_2[y*K+x], c_cpu[y*K + x] - c_gpu_2[y*K+x]);
    //             break;
    //         }
    //     }
    //     if (errors)
    //         break;
    // }
    // printf("Result: %s\n", errors?"Error":"Pass");

    // printf("test3 test =====>\n");
    // for(int y=0; y<M; y++)
    // {
    //     for(int x=0; x<K; x++)
    //     {
    //         if(fabs(c_cpu[y*K + x] - c_gpu_3[y*K+x]) > (1.0e-10))
    //         {
    //             errors = true;
    //             printf("c_cpu: %d. c_gpu_3: %d diff = %d\n", c_cpu[y*K + x], c_gpu_3[y*K+x], c_cpu[y*K + x] - c_gpu_3[y*K+x]);
    //             break;
    //         }
    //     }
    //     if (errors)
    //         break;
    // }
    // printf("Result: %s\n", errors?"Error":"Pass");

    return 0;
}

