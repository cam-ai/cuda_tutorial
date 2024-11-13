#include <stdio.h>

//统一内存行为 UM
//cudaMallocManaged()
//cudaMemPerfetchAsync(cpu)

__global__ void deviceKernel(int *a, int N)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockIdx.x * gridDim.x;
}

int main(void)
{
	printf("Hello world from CPU\n");

	float *a;

	cudaMallocManaged(&a, 1024);


	cudaDeviceSynchronize();// cpu gpu同步
	cudaDeviceReset();
	return 0;
}
