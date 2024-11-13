//https://stackoverflow.com/questions/69189880/what-is-the-difference-between-mapped-memory-and-managed-memory
#include <iostream>
#include <cassert>

__device__ float factor = 0.0;


__global__ void globalMemory()
{
    printf("device global memory : %.2f\n", factor);
    factor += 1.2;
}


int main()
{
    dim3 block(1,1);
    dim3 grid(1,1);
    float h_factor = 3.6;

    cudaSetDevice(1);

    //注意第一个参数并没有 & 取地址 它是一个符号，只需要填入变量名字即可
    cudaMemcpyToSymbol(factor, &h_factor, sizeof(float), 0, cudaMemcpyHostToDevice);

    globalMemory<<<grid, block>>>();

    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(&h_factor, factor, sizeof(float), 0, cudaMemcpyDeviceToHost);
    printf("host factor is %.2f\n", h_factor);

    //get global address
    float *pDeviceFactor;
    cudaGetSymbolAddress((void**)&pDeviceFactor, factor);
    cudaMemcpy(&h_factor, pDeviceFactor, sizeof(float), cudaMemcpyDeviceToHost);
    printf("host factor is %.2f\n", h_factor);

    //pointer attribute
    cudaPointerAttributes ptAttr;
    cudaPointerGetAttributes(&ptAttr, pDeviceFactor);
    printf("pointer attribute:device=%d, devicePointer=%p, type= %d\n",
        ptAttr.device, ptAttr.devicePointer, ptAttr.type);

    cudaDeviceReset();
    return 0;
}
