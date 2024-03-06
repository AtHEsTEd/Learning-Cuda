#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <chrono>

#define SIZE 256

using namespace std::chrono;

// Saves work during last iteration
// Volatile prevents register caching
__device__ void warp_reduce(volatile int* smem_ptr, int t)
{
	smem_ptr[t] += smem_ptr[t + 32];
	smem_ptr[t] += smem_ptr[t + 16];
	smem_ptr[t] += smem_ptr[t + 8];
	smem_ptr[t] += smem_ptr[t + 4];
	smem_ptr[t] += smem_ptr[t + 2];
	smem_ptr[t] += smem_ptr[t + 1];
}


__global__ void sum_reduction(int* vector, int* result_vector)
{
	__shared__ int partial_sum[SIZE];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// Load elements and do first add of reduction
	// Vector will be 2x no of threads, so scale i
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	partial_sum[threadIdx.x] = vector[i] + vector[i + blockDim.x];
	__syncthreads();

	// Start at 1/2 block stride and halve each time
	for (int s = blockDim.x / 2; s > 32; s >>= 1)
	{
		if (threadIdx.x < s)
		{
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	if (threadIdx.x < 32)
	{
		warp_reduce(partial_sum, threadIdx.x);
	}

	if (threadIdx.x == 0)
	{
		result_vector[blockIdx.x] = partial_sum[0];
	}
}


int main()
{
	int n = 1 << 18;
	size_t bytes = n * sizeof(int);

	int* host_v, * host_v_r;
	int* device_v, * device_v_r;

	host_v = (int*)malloc(bytes);
	host_v_r = (int*)malloc(bytes);
	cudaMalloc(&device_v, bytes);
	cudaMalloc(&device_v_r, bytes);

	std::fill_n(host_v, n, 1);
	cudaMemcpy(device_v, host_v, bytes, cudaMemcpyHostToDevice);

	const int THREADS = SIZE;
	const int BLOCKS = (int)ceil(n / static_cast<double>(THREADS) / 2);

	auto start = high_resolution_clock::now();
	sum_reduction << < BLOCKS, THREADS >> > (device_v, device_v_r);
	sum_reduction << < 1, THREADS >> > (device_v_r, device_v_r);
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(end - start);
	printf("GPU took %d microseconds\n", duration);

	cudaMemcpy(host_v_r, device_v_r, bytes, cudaMemcpyDeviceToHost);

	printf("%d\n", host_v_r[0]);
	assert(host_v_r[0] == n);

	return 0;
}