#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define DATA_SIZE 1048576

int data[DATA_SIZE];

void GenerateNumbers(int *number, int size)
{
    for (int i = 0; i < size; i++) {
        number[i] = rand() % 10;
    }
}

void printDeviceProp(const cudaDeviceProp &prop)
{
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %ld.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %ld.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %ld.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %ld.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %ld.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);

}

bool InitCUDA()
{
    int count;

    cudaGetDeviceCount(&count);

    if (count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    int i;
    for (i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printDeviceProp(prop);
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 1)
                break;
        }
    }
    if (i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.X. \n");
        return false;
    }

    cudaSetDevice(i);
    return true;
}

__global__ void sumOfSquares(int *num, int *result, clock_t *time)
{
    int sum = 0;
    int i;
    clock_t start = clock();
    for (i = 0; i < DATA_SIZE; i++) {
        sum += num[i] * num[i] * num[i];
    }
    *result = sum;
    *time = clock() - start;
}

int main()
{
    if (!InitCUDA()) {
        return 0;
    }
    printf("CUDA initialized.\n");

    GenerateNumbers(data, DATA_SIZE);

    int *gpudata, *result;
    clock_t *time;

    cudaMalloc((void **)&gpudata, sizeof(int)*DATA_SIZE);
    cudaMalloc((void **)&result, sizeof(int));
    cudaMalloc((void **)&time, sizeof(clock_t));

    cudaMemcpy(gpudata, data, sizeof(int)*DATA_SIZE, cudaMemcpyHostToDevice);

    sumOfSquares<<<1, 1, 0>>>(gpudata, result, time);

    int sum;
    clock_t time_used;
    cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_used, time, sizeof(clock_t), cudaMemcpyDeviceToHost);

    cudaFree(gpudata);
    cudaFree(result);
    cudaFree(time);

    printf("GPU sum: %d, time used: %ld\n", sum, time_used);

    sum = 0;
    time_used = clock();
    for (int i = 0; i < DATA_SIZE; i++) {
        sum += data[i] * data[i] * data[i];
    }
    time_used = clock() - time_used;

    printf("CPU sum: %d, time used: %ld\n", sum, time_used);
    return 0;
}
