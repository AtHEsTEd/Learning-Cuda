#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <chrono>

#define MASK_LENGTH 7

// Defined before all functions to be in constant memory
__constant__ int mask[MASK_LENGTH];

__global__ void convolution_1d(int* array, int* result, int n)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int r = MASK_LENGTH / 2;
	int start = tid - r;
	int temp = 0;

	for (int i = 0; i < MASK_LENGTH; i++)
	{
		// Ignore outside elements as 0s dont contribute
		if ((start + i >= 0) && (start + i < n)) temp += array[start + i] * mask[i];
	}
	result[tid] = temp;
}

void verify_result(int* array, int* mask, int* result, int n, int m)
{
	int radius = m / 2;
	int temp;
	int start;

	for (int i = 0; i < n; i++)
	{
		start = i - radius;
		temp = 0;
		for (int j = 0; j < m; j++)
		{
			if ((start + j >= 0) && (start + j < n)) temp += array[start + j] * mask[j];
		}
		assert(temp == result[i]);
	}
}

int main()
{
	int n = 1 << 22;

	// Host array
	size_t bytes = n * sizeof(int);
	int* host_array = new int[n];
	for (int i = 0; i < n; i++) host_array[i] = rand() % 100;

	// Host mask
	int* host_mask = new int[MASK_LENGTH];
	size_t bytes_m = MASK_LENGTH * sizeof(int);
	for (int i = 0; i < MASK_LENGTH; i++) host_mask[i] = rand() % 100;

	// Host result
	int* host_result = new int[n];

	// Device allocation
	int* device_array, * device_result;
	cudaMalloc(&device_array, bytes);
	cudaMalloc(&device_result, bytes);

	cudaMemcpy(device_array, host_array, bytes, cudaMemcpyHostToDevice);
	// Copy data directly to symbol for less api calls
	cudaMemcpyToSymbol(mask, host_mask, bytes_m);

	const int THREADS = 256;
	const int GRID = (n + THREADS - 1) / THREADS;

	auto start = std::chrono::high_resolution_clock::now();
	convolution_1d <<< GRID, THREADS >> > (device_array, device_result, n);
	auto end = std::chrono::high_resolution_clock::now();
	auto result = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	printf("GPU took %d microseconds \n", result);

	cudaMemcpy(host_result, device_result, bytes, cudaMemcpyDeviceToHost);
	
	start = std::chrono::high_resolution_clock::now();
	verify_result(host_array, host_mask, host_result, n, MASK_LENGTH);
	end = std::chrono::high_resolution_clock::now();
	result = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	printf("CPU took %d microseconds \n", result);

	printf("Completed\n");

	return 0;
}