#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 100
#define BLOCK_SIZE 32
#define GRID_SIZE 256

__managed__ int source[N];
__managed__ int result[1];




/** Reduce
 * 
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

int main(int argc, char **argv)
{
    int tmp = 0;
    for(int x=0; x<N; ++x) {
        //source[x] = rand()%10;
        source[x] = tmp++;
    }

    float elapsedTime;
        
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Running global reduce\n");
    cudaEventRecord(start, 0);
    for (int i = 0; i < 100; i++)
    {
        result[0] = 0;
        sum_gpu<<<GRID_SIZE, BLOCK_SIZE>>>(source, N, result);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);    
    elapsedTime /= 100.0f;      // 100 trials
    printf("Running reduce with shared mem average time elapsed: %f\n", elapsedTime);
    printf("cpu sum = %d gpu sum = %d\n", sum_cpu(source, N), result[0]);
    printf("source [0] = %d\n", source[0]);


        
    return 0;
}