#include <pthread.h>
#include <iostream>

const int num_loops = 1024;
const int nBLK = 2;
const int nTPB = 128;
const int num_pthreads = 2;
const int ds = 1048576;

__global__ void k(int *data, int *data_b, int *data_c, size_t N)
{
    for (size_t idx = blockIdx.x*blockDim.x+threadIdx.x; idx < N; idx += gridDim.x*blockDim.x) {
        data[idx]++;
        data_b[idx]++;
        data_c[idx]++;
    }
        
}

struct threadInfo
{
    int *data;
    int *data_b;
    int *data_c;
    size_t my_N;
    cudaStream_t s;
    int test;
};

void *threadFunc(void* arg)
{
    struct threadInfo* threadInfoStruct;
    threadInfoStruct = (struct threadInfo*) arg;
    for (int i = 0; i < num_loops; i++){
        k<<<nBLK, nTPB, 0, threadInfoStruct->s>>>(threadInfoStruct->data,threadInfoStruct->data_b,threadInfoStruct->data_c, threadInfoStruct->my_N);
        cudaStreamSynchronize(threadInfoStruct->s);
        threadInfoStruct->test = (threadInfoStruct->data)[0];
    }
    pthread_exit(NULL);
}

//#define USE_STREAM_ATTACH
int main()
{

    int *data[num_pthreads];
    int *data_b[num_pthreads];
    int *data_c[num_pthreads];
    cudaStream_t streams[num_pthreads];
    for (int i = 0; i < num_pthreads; i++){
        cudaMallocManaged(data+i, sizeof(int)*ds,   cudaMemAttachHost);
        cudaMallocManaged(data_b+i, sizeof(int)*ds, cudaMemAttachHost);
        cudaMallocManaged(data_c+i, sizeof(int)*ds, cudaMemAttachHost);
        for (int j = 0; j < ds; j++) {
            data[i][j] = 0;
            data_b[i][j] = 0;
            data_c[i][j] = 0;
        }


        cudaStreamCreate(streams+i);
#ifdef USE_STREAM_ATTACH
        cudaStreamAttachMemAsync(streams[i], data[i],   0, cudaMemAttachSingle);
        cudaStreamAttachMemAsync(streams[i], data_b[i], 0, cudaMemAttachSingle);
        cudaStreamAttachMemAsync(streams[i], data_c[i], 0, cudaMemAttachSingle);
        cudaStreamSynchronize(streams[i]);
#endif
    }

    threadInfo ti[num_pthreads];
    pthread_t threads[num_pthreads];
    for (int i = 0; i < num_pthreads; i++){
        ti[i].data = data[i];
        ti[i].data_b = data_b[i];
        ti[i].data_c = data_c[i];
        ti[i].my_N = ds;
        ti[i].s = streams[i];
        int rs = pthread_create(threads+i, NULL, threadFunc, (void *) (ti+i));
        if (rs != 0)
            std::cout << "pthread_create error: " << rs << std::endl;
    }

    for (int i = 0; i < num_pthreads; i++){
        int rs = pthread_join(threads[i], NULL);
        if (rs != 0)
            std::cout << "pthread_join error: " << rs << std::endl;
    }

    for (int i = 0; i < num_pthreads; i++)
        std::cout << "thread: " << i << " expected value: " << num_loops << " final value: " << ti[i].test << std::endl;
    return 0;
}
