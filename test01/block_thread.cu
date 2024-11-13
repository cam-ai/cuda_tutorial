#include <stdio.h>

__global__ void helloFromGPU(void) 
{
	if (threadIdx.x == 2 && blockIdx.x ==1 )
		printf("successfull\n");
	else
		printf("Hello world from GPU\n");
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
