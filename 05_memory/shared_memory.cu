//https://stackoverflow.com/questions/69189880/what-is-the-difference-between-mapped-memory-and-managed-memory
#include <iostream>
#include <cassert>

/*

https://zhuanlan.zhihu.com/p/445373116

常量内存
常量变量用__constant__修饰符进行修饰，它们必须在全局空间内和所有内核函数之外进行声明，对同一编译单元中的内核函数都是可见的。常量变量存储在常量内存中，内核函数只能从常量内存中读取数据，常量内存必须在host端代码中使用下面的函数来进行初始化：

cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src,size_t count， kind);
下面的例子展示了如何声明常量内存并与之进行数据交换：

__constant__ float const_data[256];
float data[256];
cudaMemcpyToSymbol(const_data, data, sizeof(data));
cudaMemcpyFromSymbol(data, const_data, sizeof(data));
常量内存适合用于线程束中的所有线程都需要从相同的内存地址中读取数据的情况，比如所有线程都需要的常量参数，每个GPU只可以声明不超过64KB的常量内存
*/

__shared__ float g_shared;

extern __shared__ int dynamic_array[];

__global__ void dynamic_sm_kernel()
{
    dynamic_array[threadIdx.x] = threadIdx.x;
    printf("access dynamic_array in kernel, dynamic_array[%d]=%d\n",
        threadIdx.x,
        dynamic_array[threadIdx.x]);
}

__global__ void kernel_1()
{
    __shared__ float k1_shared;
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (blockIdx.x == 0 && id == 0) {
        k1_shared = 5.0f;
    }
    if (blockIdx.x ==1 && id == 16) {
        k1_shared = 6.0f;
    }
    __syncthreads();
    printf("access local shared in kernel1, k1_shared = %f, blockIdx = %d, threadIdx=%d, threadId=%d\n", k1_shared, blockIdx.x, threadIdx.x, id);


}

__global__ void kernel_2()
{
    g_shared = 0.0f;
    printf("access global shared in kernel 2 g_shared = %f\n", g_shared);
}



int main()
{
    dim3 block(16);
    dim3 grid(2);
    float h_factor = 2.3;

    kernel_1<<<grid, block>>>();
    kernel_2<<<grid, block>>>();

    dynamic_sm_kernel<<<grid, block, block.x * sizeof(int)>>>();

    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}
