#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../scope_timer.hpp"

#define N 1024
#define BLOCK_SIZE 32
#define GRID_SIZE 256

__managed__ int source[N];
__managed__ int result[1];
__managed__ int output_buffer[N];




/** Reduce
 * https://blog.csdn.net/qq_45788429/article/details/133950846
*/


__global__ void sum_gpu(const int *in, int count, int *out)
{
    __shared__ int ken[BLOCK_SIZE];
    // grid loop
    int shared_tmp = 0;
    //读数据的过程中，顺便做一下加法，再放入shared memory中
    for (int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < count; idx+= blockDim.x * gridDim.x)
    {
        shared_tmp += in[idx];
    }

    ken[threadIdx.x] = shared_tmp;
    __syncthreads();

    int tmp = 0;
    for (int total_threads = BLOCK_SIZE/2; total_threads >= 1; total_threads /=2)
    {
        if (threadIdx.x < total_threads)
        {
            tmp = ken[threadIdx.x] + ken[threadIdx.x+total_threads];
        }
        __syncthreads();

        if (threadIdx.x < total_threads)
        {
            ken[threadIdx.x] = tmp;
        }
    }

    //block_sum -> share memory[0]
    if (blockIdx.x * blockDim.x < count)
    {
        if (threadIdx.x == 0)
        {
            //out[0] += ken[0]; // conficts
            atomicAdd(out, ken[0]); // memory space write modifiy read
        }
    }

}

int sum_cpu(const int *in, int count)
{
    int sum = 0;
    for (int i = 0 ; i < count; i++) {
        sum += in[i];
    }
    return sum;
}

__global__ void gpu_get_sum_1(const int *g_idata, int count, int *g_odata)
{
    extern __shared__ int sdata[];
    //each thread loads one element from global to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    //do reduction in shared memory
    for (unsigned int s = 1; s < blockDim.x; s*=2)
    {
        if (tid % (2*s) == 0)
        {
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    //write result for this block to global memory
    if (tid ==0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }

}


__global__ void gpu_get_sum_2(const int *g_idata, int count, int *g_odata)
{
    extern __shared__ int sdata[];
    //each thread loads one element from global to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    //do reduction in shared memory
    for (unsigned int s = 1; s < blockDim.x; s*=2)
    {
        if (tid % (2*s) == 0)
        {
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    //write result for this block to global memory
    if (tid ==0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }

}


int main(int argc, char **argv)
{
    int tmp = 0;
    for(int x=0; x<N; ++x) {
        source[x] = rand()%10;
        //source[x] = tmp++;
    }

    int cpu_sum = 0;
    {
        ScopeTimer<TimerPlatForm::CPU> timer("CPU Get Sum Cost  ");
        cpu_sum = sum_cpu(source, N);
    }
    printf("cpu get sum = %d\n", cpu_sum);

    int numThreadsPerBlock=BLOCK_SIZE;
    int numInputElements = N;
    int numOutputElements = 0;
    numOutputElements = numInputElements / (numThreadsPerBlock / 2);
    if (numOutputElements % (numThreadsPerBlock/2))
    {
        numOutputElements++;
    }

    {
        ScopeTimer<TimerPlatForm::GPU> timer("GPU Get Sum Cost  Kun ");
        sum_gpu<<<GRID_SIZE, BLOCK_SIZE>>>(source, N, result);
    }
    cudaDeviceSynchronize();// cpu gpu同步
    printf("gpu get sum = %d\n", result[0]);
    

    int gpu_sum = 0;
    {
        
        ScopeTimer<TimerPlatForm::GPU> timer("GPU Get Sum Cost  ");
        gpu_get_sum_1<<<((N+BLOCK_SIZE-1)/BLOCK_SIZE/2), BLOCK_SIZE, BLOCK_SIZE*sizeof(int)>>>(source, N, output_buffer);
        cudaDeviceSynchronize();// cpu gpu同步
        gpu_get_sum_1<<<1, BLOCK_SIZE, BLOCK_SIZE*sizeof(int)>>>(output_buffer, N/BLOCK_SIZE/2, result);

        // for (int ii = 1 ; ii < numOutputElements; ii++)
        // {
        //     //output_buffer[0] += output_buffer[ii];
        //    // printf("data = %d\n", output_buffer[ii]);
        // }

    }
    cudaDeviceSynchronize();// cpu gpu同步
    printf("gpu get sum = %d\n", result[0]);



        
    return 0;
}