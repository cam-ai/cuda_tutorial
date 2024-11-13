#include <stdio.h>

__global__ void helloFromGPU(void) 
{
	printf("successfull %d \n", (threadIdx.x) *(blockIdx.x +1) -1 );
}

int main(void)
{
	printf("Hello world from CPU\n");

	helloFromGPU<<<2, 10>>>(); // <<<number of blocks, number of threads per block>>>
				   // <<<blockIdx, threadIdx.x>>>

	cudaDeviceSynchronize();// cpu gpu同步
	cudaDeviceReset();
	return 0;
}
