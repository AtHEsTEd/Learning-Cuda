#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <chrono>

#define SIZE 128

using namespace std::chrono;
using namespace cooperative_groups;

__device__ int reduce_sum(thread_group g, int* temp, int val) 
{
	int lane = g.thread_rank();

	// Each thread adds its partial sum[i] to sum[lane+i]
	for (int i = g.size() / 2; i > 0; i >>= 1) {
		temp[lane] = val;
		// wait for all threads to store
		g.sync();
		if (lane < i) 
		{
			val += temp[lane + i];
		}
		// wait for all threads to load
		g.sync();
	}
	// note: only thread 0 will return full sum
	return val;
}

__device__ int thread_sum(int* input, int n) 
{
	int sum = 0;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = tid; i < n / 4; i += blockDim.x * gridDim.x) 
	{
		// Cast as int4 
		int4 in = ((int4*)input)[i];
		sum += in.x + in.y + in.z + in.w;
	}
	return sum;
}

__global__ void sum_reduction(int* sum, int* input, int n) 
{
	// Create partial sums from the array
	int my_sum = thread_sum(input, n);

	// Dynamic shared memory allocation
	extern __shared__ int temp[];

	// Identifier for a TB
	auto g = this_thread_block();

	// Reudce each TB
	int block_sum = reduce_sum(g, temp, my_sum);

	// Collect the partial result from each TB
	if (g.thread_rank() == 0) {
		atomicAdd(sum, block_sum);
	}
}


int main() {
	// Vector size
	int n = 1 << 13;
	size_t bytes = n * sizeof(int);

	// Original vector and result vector
	int* sum;
	int* data;

	// Allocate using unified memory
	cudaMallocManaged(&sum, sizeof(int));
	cudaMallocManaged(&data, bytes);

	// Initialize vector
	std::fill_n(data, n, 1);

	// TB Size
	int TB_SIZE = SIZE;

	// Grid Size (cut in half)
	int GRID_SIZE = (n + TB_SIZE - 1) / TB_SIZE;

	// Call kernel with dynamic shared memory (Could decrease this to fit larger data)
	auto start = high_resolution_clock::now();
	sum_reduction << <GRID_SIZE, TB_SIZE, n * sizeof(int) >> > (sum, data, n);
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(end - start);
	printf("GPU took %d microseconds \n", duration);

	// Synchronize the kernel
	cudaDeviceSynchronize();

	printf("%d\n", sum[0]);

	printf("COMPLETED SUCCESSFULLY\n");

	return 0;
}