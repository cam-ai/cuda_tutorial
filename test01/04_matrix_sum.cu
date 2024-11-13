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

__global__ void sumMatrixOnGPU15D(int *MatA, int *MatB, int *MatC, int nx, int ny)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = blockIdx.y; //threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * nx + ix;

	if (ix < nx && iy < ny)
		MatC[idx] = MatA[idx] + MatB[idx];
}

__global__ void sumMatrixOnGPU2D(int *MatA, int *MatB, int *MatC, int nx, int ny)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * nx + ix;

	if (ix < nx && iy < ny)
		MatC[idx] = MatA[idx] + MatB[idx];
}

__global__ void sumMatrixOnGPU1D(int *MatA, int *MatB, int *MatC, int nx, int ny)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	//unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	//unsigned int idx = iy * nx + ix;

	if (ix < nx)
	{
		for(int iy = 0; iy < ny; iy++)
		{
			int idx = iy * nx + ix;
			MatC[idx] = MatA[idx] + MatB[idx];

		}
	}
		
}

__global__ void printThreadIndex(int *A, const int nx, const int ny)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * nx + ix;

	printf("Thread_id (%2d, %2d) block_id (%2d %2d) coordinate (%2d %2d) global index %2d, value %2d DimSize(%2d,%2d)\n",
		threadIdx.x, threadIdx.y,
		blockIdx.x, blockIdx.y,
		ix, iy,
		idx, A[idx],
		blockDim.x, blockDim.y
	);
}

int main(int argc, char **argv)
{
	if (argc != 3) {
		printf("usage: app nx ny\n");
		return -1;
	}

	int block_x = atoi(argv[1]);
	int block_y = atoi(argv[2]);


		
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
	int *d_A;
	int *d_B;
	int *d_C;
	cudaMalloc((void**)&d_A, nBytes);
	cudaMalloc((void**)&d_B, nBytes);
	cudaMalloc((void**)&d_C, nBytes);

	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

	//set up execution configuration
	dim3 block(block_x, block_y);
	dim3 grid((nx + block.x - 1)/block.x, (ny + block.y-1)/block.y);
	printf("Thread config grid (%02d %02d) block (%2d, %2d)\n", grid.x, grid.y, block.x, block.y);

	cudaEvent_t event_start, event_end;
	float elapsed = 0.0f;
	cudaEventCreate(&event_start);
    cudaEventCreate(&event_end);

	cudaEventRecord(event_start);
	sumMatrixOnGPU2D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
	cudaEventRecord(event_end);
	cudaEventSynchronize(event_end);
    cudaEventElapsedTime(&elapsed, event_start, event_end);
	printf("duration 2D: %f\n", elapsed);


	dim3 block1d(block_x, 1);
	dim3 grid1d((nx + block1d.x - 1)/block1d.x, 1);
	cudaEventRecord(event_start);
	sumMatrixOnGPU1D<<<grid1d, block1d>>>(d_A, d_B, d_C, nx, ny);
	cudaEventRecord(event_end);
	cudaEventSynchronize(event_end);
    cudaEventElapsedTime(&elapsed, event_start, event_end);
	printf("duration 1D: %f\n", elapsed);


	dim3 block15d(block_x, 1);
	dim3 grid15d((nx + block.x - 1)/block.x, ny);
	cudaEventRecord(event_start);
	sumMatrixOnGPU15D<<<grid15d, block15d>>>(d_A, d_B, d_C, nx, ny);
	cudaEventRecord(event_end);
	cudaEventSynchronize(event_end);
    cudaEventElapsedTime(&elapsed, event_start, event_end);
	printf("duration 1.5D: %f\n", elapsed);


	

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
