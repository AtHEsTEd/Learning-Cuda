#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <chrono>

#define SIZE 1024

using namespace std::chrono;

__global__ void sum_reduction(int* vector, int* vector_result)
{
	__shared__ int partial_sum[SIZE];
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	partial_sum[threadIdx.x] = vector[id];
	__syncthreads();

	for (int s = 1; s < blockDim.x; s *= 2)
	{
		int index = 2 * s * threadIdx.x;
		if (index < blockDim.x)
		{
			partial_sum[index] += partial_sum[index + s];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		vector_result[blockIdx.x] = partial_sum[0];
	}
}

void init_vector(int* vector, int n)
{
	for (int i = 0; i < n; i++)
	{
		vector[i] = 1;
	}
}

int main()
{
	int n = 1 << 20;
	size_t bytes = n * sizeof(int);

	int *host_v, *host_v_r;
	int *device_v, *device_v_r;

	host_v = (int*)malloc(bytes);
	host_v_r = (int*)malloc(bytes);
	cudaMalloc(&device_v, bytes);
	cudaMalloc(&device_v_r, bytes);

	std::fill_n(host_v, n, 1);
	cudaMemcpy(device_v, host_v, bytes, cudaMemcpyHostToDevice);

	const int THREADS = SIZE;
	const int BLOCKS = (int)ceil(n / THREADS);

	auto start = high_resolution_clock::now();
	sum_reduction << < BLOCKS, THREADS >> > (device_v, device_v_r);
	sum_reduction << < 2, THREADS >> > (device_v_r, device_v_r);
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(end - start);
	printf("GPU took %d microseconds\n", duration);

	cudaMemcpy(host_v_r, device_v_r, bytes, cudaMemcpyDeviceToHost);

	printf("%d\n", host_v_r[0]);
	assert(host_v_r[0] == n);

	return 0;
}