#include <stdio.h>

__global__ void helloFromGPU(void) //__global__ 关键字
{
	printf("Hello world from GPU\n");
}

int main(void)
{
	printf("Hello world from CPU\n");

	// 内核调用
	helloFromGPU<<<2, 10>>>(); // <<<number of blocks, number of threads per block(1024 the biggest )>>>
				   // <<<blockIdx, threadIdx.x>>>
				   // <<<grid, block>>

	cudaDeviceSynchronize();// cpu gpu同步
	// 释放所有与当前进程相关的GPU资源
	cudaDeviceReset();
	return 0;
}
