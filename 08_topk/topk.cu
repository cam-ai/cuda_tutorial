#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "../scope_timer.hpp"

#define N 100000000
#define BLOCK_SIZE 32
#define GRID_SIZE 256
#define topk    20
//#define INT_MIN 0

__managed__ int source[N];
__managed__ int _1_pass_result[topk * GRID_SIZE];
__managed__ int gpu_result[topk];
int cpu_result[topk];




/** Reduce
 * 
*/

__device__ __host__ void insert_value(int *array, int k, int data)
{
    for(int i =0; i < k; i++)
    {
        if (array[i] == data){
            return;
        }
    }

    if (data < array[k -1])
    {
        return;
    }

    for (int i = k - 2; i >=0; i--)
    {
        if (data > array[i])
        {
            array[i+1] = array[i];
        }
        else
        {
            array[i+1] = data;
            return;
        }
    }
    array[0] = data;
}


__global__ void gpu_topk(const int *in, int *output, int count, int k)
{
    __shared__ int ken[BLOCK_SIZE*topk];
    int top_array[topk];
    for (int i = 0; i < topk; i++)
    {
        top_array[i] = INT_MIN;
    }

    for (int idx = threadIdx.x+blockDim.x*blockIdx.x; idx < count; idx += gridDim.x * blockDim.x)
    {
        insert_value(top_array, topk, in[idx]);
    }

    for (int i = 0; i < topk; i++)
    {
        ken[topk * threadIdx.x + i] = top_array[i];
    }
    __syncthreads();

    for (int i = BLOCK_SIZE/2; i >= 1; i/=2)
    {
        if (threadIdx.x < i)
        {
            for (int m = 0; m < topk; m++)
            {
                insert_value(top_array, topk, ken[topk * (threadIdx.x + i) +m]);
            }
        }
        __syncthreads();
        if (threadIdx.x < i)
        {
            for (int m=0; m < topk; m++)
            {
                ken[topk * threadIdx.x+m] = top_array[m];
            }
        }
        __syncthreads();
    }

    if (blockIdx.x * blockDim.x < count)
    {
        if (threadIdx.x == 0) //cat be any thread 
        {
            for (int i = 0; i < topk; i++)
            {
                output[topk * blockIdx.x + i] = ken[i];
            }
        }
    }

}

int cpu_topk(const int *in, int *output, int count, int k)
{
    //std::sort(in,)
    for(int i =0; i < count; i++)
    {
        // for(int j = 0; j < k-1; j++)
        // {
        //     if(in[i] >= output[j])
        //     {
        //         output[j+1] = output[j];
        //         output[j] = in[i];
        //     }
        // }.

        insert_value(output, k, in[i]);
    }
    return 0;
}

int main(int argc, char **argv)
{
    int tmp = 0;
    for(int x=0; x<N; ++x) {
        //source[x] = rand()%1024;
        source[x] = rand();
        //source[x] = tmp++;
    }

    {
        ScopeTimer<TimerPlatForm::GPU> timer("GPU Get TopK  : ");

        gpu_topk<<<GRID_SIZE, BLOCK_SIZE>>>(source, _1_pass_result,  N, topk);
        gpu_topk<<<1, BLOCK_SIZE>>>(_1_pass_result, gpu_result,  topk * GRID_SIZE, topk);

    }

    {
        ScopeTimer<TimerPlatForm::CPU> timer("CPU Get TopK  : ");
        cpu_topk(source, cpu_result, N, topk);
    }


    for (int i = 0 ; i < topk; i++)
    {
        printf("CPU top%2d : %8d \tGPU%2d : %8d\n", i+1, cpu_result[i], i+1, gpu_result[i]);
        if (cpu_result[i] != gpu_result[i])
        {
            printf("failed\n");
            break;
        }
    }


        
    return 0;
}