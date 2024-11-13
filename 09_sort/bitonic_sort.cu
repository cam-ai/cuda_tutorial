/**
 * https://www.jianshu.com/p/137e5cee76c7
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "../scope_timer.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 512

__global__ void bitonic_sort_step(int *dev_values, int j, int k) {
    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i^j;

    /* The threads with the lowest ids sort the array. */
    if ((ixj)>i) {
        if ((i&k)==0) {
            /* Sort ascending */
            if (dev_values[i]>dev_values[ixj]) {
                /* exchange(i,ixj); */
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
        if ((i&k)!=0) {
            /* Sort descending */
            if (dev_values[i]<dev_values[ixj]) {
                /* exchange(i,ixj); */
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
    }
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(int *values) {
    int *dev_values;
    size_t size = N * sizeof(int);

    cudaMalloc((void**) &dev_values, size);
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

    dim3 blocks(N/2,1);    /* Number of blocks   */
    dim3 threads(256,1);  /* Number of threads  */

    int j, k;
    static int iter_times{0};
    /* Major step */
    for (k = 2; k <= N; k <<= 1) {
        /* Minor step */
        for (j=k>>1; j>0; j=j>>1) {
            bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
            printf("iter time is %d,k= %d, j=%d\n",++iter_times,k,j);
        }
    }
    cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_values);
}

int main() {
    int h[N];
    for (int i = 0; i < N; i++) {
        h[i] = N-i;
    }

    bitonic_sort(h);

    for (int i = 0; i < N; i++) {
        printf("%d ", h[i]);
    }
    printf("\n");

    return 0;
}
