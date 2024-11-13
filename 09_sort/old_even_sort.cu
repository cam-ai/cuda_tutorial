#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "../scope_timer.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 512

__global__ void gpu_odd_even_sort(int *d, int n, int phase) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (phase % 2 == 0) { // even phase
        if (index < n / 2) {
            int i = 2 * index;
            if (d[i] > d[i + 1]) {
                // swap elements
                int temp = d[i];
                d[i] = d[i + 1];
                d[i + 1] = temp;
            }
        }
    } else { // odd phase
        if (index < (n - 1) / 2) {
            int i = 2 * index + 1;
            if (d[i] > d[i + 1]) {
                // swap elements
                int temp = d[i];
                d[i] = d[i + 1];
                d[i + 1] = temp;
            }
        }
    }
}


int main() {
    int h[N];
    int *d;

    // Initialize host array
    for (int i = 0; i < N; i++) {
        // h[i] = rand() % N;
        h[i] = N-i;
    }

    // 打印结果
    printf("======初始数据========\n");
    for (size_t i = 0; i < N; i++)
    {
        printf("%d\t",h[i]);
    }
    printf("\n");

    // Allocate device memory
    cudaMalloc((void**)&d, sizeof(int) * N);

    // Copy host array to device array
    cudaMemcpy(d, h, sizeof(int) * N, cudaMemcpyHostToDevice);

    // Launch kernel
    for (int i = 0; i < N; i++) {
        gpu_odd_even_sort<<<N / 256, 256>>>(d, N, i);
        cudaDeviceSynchronize();
    }

    // Copy back to host
    cudaMemcpy(h, d, sizeof(int) * N, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d);

    // 打印结果
    printf("======结果========\n");
    for (size_t i = 0; i < N; i++)
    {
        printf("%d\t",h[i]);
    }
    printf("\n");

    return 0;
}
