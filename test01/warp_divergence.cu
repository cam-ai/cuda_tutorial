#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "cuda.h"

void print_matrix(int *C, const int nx, const int ny)
{
	for (int iy = 0; iy < ny; iy++) {
		for (int ix = 0; ix < nx; ix++) {
			printf("%3d\t", C[iy*nx+ix]);
		}
		printf("\n");
	}
	printf("\n");
	return;
}

__global__ void mathKernel2(float *c)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	float a ,b;
	a = b = 0.0f;
	// warpSize GPU 中通常为32, 有分支，也只会发生在同一个warp中，所以不影响效率
	if ((tid % warpSize) % 2 ==0) {
		a = 100.0f;
	} else {
		b = 200.0f;
	}
	c[tid] = a + b;
}

__global__ void mathKernel_with_divergency(float *c)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	float ia, ib;
	ia = ib = 0.0f;
	bool ipred = (tid %2 == 0);

	if (tid %2 == 0) {
		ia = 100.f;
	} else {
		ib = 200.f;
	}
	c[tid] = ia + ib;
}

__global__ void mathKernel_without_divergency(float *c)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	float ia, ib;
	ia = ib = 0.0f;
	bool ipred = (tid %2 == 0); //每个线程都会执行， 避免tid因为id的不一样，产生线程束分支
	//下面的代码已经脱离了与线程id的关联关系，不会产生divergency
	if (ipred) {
		ia = 100.f;
	}

	if (!ipred) {
		ib = 200.f;
	}
	c[tid] = ia + ib;
}


int main(int argc, char **argv)
{
	if (argc != 2) {
		printf("usage: app nx ny\n");
		return -1;
	}

	int block_x = atoi(argv[1]);


		
	int nx = 1 << 14, ny = 1<<14;
	int nxy = nx * ny;
	int nBytes = nxy * sizeof(int);

	int *h_A;
	h_A = static_cast<int*>(malloc(nBytes));
	int *h_B;
	h_B = static_cast<int*>(malloc(nBytes));
	int *gpuRef;
	gpuRef = static_cast<int*>(malloc(nBytes));

	for(int i = 0; i < nxy; i++) {
		h_A[i] = i;
		h_B[i] = i + 1;
	}
	//print_matrix(h_A, nx, ny);
	float *d_A;
	int *d_B;
	int *d_C;
	cudaMalloc((void**)&d_A, nBytes);
	cudaMalloc((void**)&d_B, nBytes);
	cudaMalloc((void**)&d_C, nBytes);

	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

	cudaEvent_t event_start, event_end;
	float elapsed = 0.0f;
	cudaEventCreate(&event_start);
    cudaEventCreate(&event_end);


	dim3 block1d(block_x, 1);
	dim3 grid1d((nx + block1d.x - 1)/block1d.x, 1);
	cudaEventRecord(event_start);
	mathKernel_without_divergency<<<grid1d, block1d>>>(d_A);
	cudaEventRecord(event_end);
	cudaEventSynchronize(event_end);
    cudaEventElapsedTime(&elapsed, event_start, event_end);
	printf("duration  mathKernel_without_divergency 1D: %f\n", elapsed);


	cudaEventRecord(event_start);
	mathKernel_with_divergency<<<grid1d, block1d>>>(d_A);
	cudaEventRecord(event_end);
	cudaEventSynchronize(event_end);
    cudaEventElapsedTime(&elapsed, event_start, event_end);
	printf("duration  mathKernel_with_divergency 1D: %f\n", elapsed);


	

	cudaEventDestroy(event_start);
	cudaEventDestroy(event_end);


	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();// cpu gpu同步
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);
	free(gpuRef);
	cudaDeviceReset();
	return 0;
}
