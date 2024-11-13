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

__constant__ float factor;


__global__ void constantMemory()
{
    printf("Get const memory : %.2f\n", factor);
}


int main()
{
    dim3 block(8,1);
    dim3 grid(1,1);
    float h_factor = 2.3;

    //注意第一个参数并没有 & 取地址 它是一个符号，只需要填入变量名字即可
    cudaMemcpyToSymbol(factor, &h_factor, sizeof(float), 0, cudaMemcpyHostToDevice);

    constantMemory<<<grid, block>>>();

    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}
