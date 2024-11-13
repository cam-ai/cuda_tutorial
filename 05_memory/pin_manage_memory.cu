//https://stackoverflow.com/questions/69189880/what-is-the-difference-between-mapped-memory-and-managed-memory
#include <iostream>
#include <cassert>

__global__
void kernel(char* __restrict__ data, int pagesize, int numpages){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < numpages){
        data[tid * pagesize] += 1;
    }
}

int main(){
    const int pagesize = 4096;
    const int numpages = 1024 * 64;
    const int bytes = pagesize * numpages;
    cudaError_t status = cudaSuccess;
    float elapsed = 0.0f;
    const int iterations = 5;

    char* devicedata; 
    status = cudaMalloc(&devicedata, bytes);
    assert(status == cudaSuccess);

    char* pinneddata; 
    status = cudaMallocHost(&pinneddata, bytes);
    assert(status == cudaSuccess);

    char* manageddata;
    status = cudaMallocManaged(&manageddata, bytes);
    assert(status == cudaSuccess);

    status = cudaMemPrefetchAsync(manageddata, bytes, cudaCpuDeviceId);
    //status = cudaMemPrefetchAsync(manageddata, bytes, 0);
    assert(status == cudaSuccess);

    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    for(int iteration = 0; iteration < iterations; iteration++){
        cudaEventRecord(event1);
        kernel<<<numpages / 256, 256>>>(pinneddata, pagesize, numpages);
        cudaEventRecord(event2);
        status = cudaEventSynchronize(event2);
        assert(status == cudaSuccess);
        cudaEventElapsedTime(&elapsed, event1, event2);
        
        float bandwith = (numpages / elapsed) * 1000.0f / 1024.f / 1024.f;
        std::cerr << "pinned: " << elapsed << ", throughput " << bandwith << " GB/s" << "\n";
    }

    for(int iteration = 0; iteration < iterations; iteration++){
        cudaEventRecord(event1);
        kernel<<<numpages / 256, 256>>>(manageddata, pagesize, numpages);
        cudaEventRecord(event2);
        status = cudaEventSynchronize(event2);
        assert(status == cudaSuccess);
        cudaEventElapsedTime(&elapsed, event1, event2);

        float bandwith = (numpages / elapsed) * 1000.0f / 1024.f / 1024.f;
        std::cerr << "managed: " << elapsed << ", throughput " << bandwith << " MB/s" << "\n";

        //status = cudaMemPrefetchAsync(manageddata, bytes, cudaCpuDeviceId);
        assert(status == cudaSuccess);     
    }

    for(int iteration = 0; iteration < iterations; iteration++){
        cudaEventRecord(event1);
        kernel<<<numpages / 256, 256>>>(devicedata, pagesize, numpages);
        cudaEventRecord(event2);
        status = cudaEventSynchronize(event2);
        assert(status == cudaSuccess);
        cudaEventElapsedTime(&elapsed, event1, event2);
        
        float bandwith = (numpages / elapsed) * 1000.0f / 1024.f / 1024.f;
        std::cerr << "device: " << elapsed << ", throughput " << bandwith << " MB/s" << "\n";
    }

    cudaFreeHost(pinneddata);
    cudaFree(manageddata);
    cudaFree(devicedata);
    cudaEventDestroy(event1);
    cudaEventDestroy(event2);

}
