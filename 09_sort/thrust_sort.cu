#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "../scope_timer.hpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#define N 512

int main() {
    // Generate random numbers on the host
    thrust::host_vector<int> h(N);
    for (int i = 0; i < N; i++) {
        h[i] = rand() % N;
    }

    // Copy data to device
    thrust::device_vector<int> d = h;

    // Sort data on the device (uses Thrust's CUDA implementation of quicksort)
    thrust::sort(d.begin(), d.end());

    // Copy data back to host
    thrust::copy(d.begin(), d.end(), h.begin());

    // Print sorted data
    for (int i = 0; i < N; i++) {
        printf("%d ", h[i]);
    }
    printf("\n");

    return 0;
}
