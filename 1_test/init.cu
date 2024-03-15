#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DATA_SIZE 1048576

int data[DATA_SIZE];

void GenerateNumbers(int *number, int size)
{
    for (int i = 0; i < size; i++) {
        number[i] = rand() % 10;
    }
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

__global__ void sumOfSquares(int *num, int *result)
{
    int sum = 0;
    int i;
    for (i = 0; i < DATA_SIZE; i++) {
        sum += num[i] * num[i] * num[i];
    }
    *result = sum;
}

int main()
{
    if (!InitCUDA()) {
        return 0;
    }
    printf("CUDA initialized.\n");

    GenerateNumbers(data, DATA_SIZE);

    int *gpudata, *result;

    cudaMalloc((void **)&gpudata, sizeof(int)*DATA_SIZE);
    cudaMalloc((void **)&result, sizeof(int));

    cudaMemcpy(gpudata, data, sizeof(int)*DATA_SIZE, cudaMemcpyHostToDevice);

    sumOfSquares<<<1, 1, 0>>>(gpudata, result);

    int sum;
    cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(gpudata);
    cudaFree(result);

    printf("GPU sum: %d\n", sum);

    sum = 0;
    for (int i = 0; i < DATA_SIZE; i++) {
        sum += data[i] * data[i] * data[i];
    }

    printf("CPU sum: %d\n", sum);
    return 0;
}
