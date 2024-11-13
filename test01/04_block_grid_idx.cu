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

int main(void)
{
	int nx = 8, ny = 6;
	int nxy = nx * ny;
	int nBytes = nxy * sizeof(int);

	int *h_A;
	h_A = static_cast<int*>(malloc(nBytes));
	for(int i = 0; i < nxy; i++) {
		h_A[i] = i;
	}
	print_matrix(h_A, nx, ny);
	int *d_A;
	cudaMalloc((void**)&d_A, nBytes);
	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

	//set up execution configuration
	dim3 block(4, 2);
	dim3 grid((nx + block.x - 1)/block.x, (ny + block.y-1)/block.y);
	printf("grid (%02d %02d)\n", grid.x, grid.y);

	printThreadIndex<<<grid, block>>>(d_A, nx, ny);



	cudaDeviceSynchronize();// cpu gpu同步
	cudaFree(d_A);
	free(h_A);
	cudaDeviceReset();
	return 0;
}
