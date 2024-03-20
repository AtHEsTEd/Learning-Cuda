#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <chrono>
#include <stdio.h>

#define MASK_LENGTH 7

// Defined before all functions to be in constant memory
__constant__ int mask[MASK_LENGTH];

__global__ void convolution_1d(int* array, int* result, int n)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ int smem_array[];

	// Load into shared memory, already offset due to padding
	smem_array[threadIdx.x] = array[tid];
	
	__syncthreads();

	int temp = 0;

	for (int i = 0; i < MASK_LENGTH; i++)
	{
		if (threadIdx.x + i >= blockDim.x) temp += array[tid + i] * mask[i];
		else temp += smem_array[threadIdx.x + i] * mask[i];
	}

	result[tid] = temp;
}

void verify_result(int* array, int* mask, int* result, int n) {

	int temp;
	for (int i = 0; i < n; i++) 
	{
		temp = 0;
		for (int j = 0; j < MASK_LENGTH; j++) 
		{
			temp += array[i + j] * mask[j];
		}
		assert(temp == result[i]);
	}
}

int main()
{
	int n = 1 << 32;
	size_t bytes = n * sizeof(int);

	// Host mask
	int* host_mask = new int[MASK_LENGTH];
	size_t bytes_m = MASK_LENGTH * sizeof(int);
	for (int i = 0; i < MASK_LENGTH; i++) host_mask[i] = rand() % 100;

	// Padded array
	int r = MASK_LENGTH / 2;
	int n_p = n + r * 2;
	size_t bytes_p = n_p * sizeof(int);
	int* host_array = new int[n_p];

	for (int i = 0; i < n_p; i++)
	{
		if (i < r || i >= n + r) host_array[i] = 0;
		else host_array[i] = rand() % 100;
	}

	// Host result
	int* host_result = new int[n];

	// Device allocation
	int* device_array, * device_result;
	cudaMalloc(&device_array, bytes_p);
	cudaMalloc(&device_result, bytes);

	cudaMemcpy(device_array, host_array, bytes_p, cudaMemcpyHostToDevice);
	// Copy data directly to symbol for less api calls
	cudaMemcpyToSymbol(mask, host_mask, bytes_m);

	const int THREADS = 256;
	const int GRID = (n + THREADS - 1) / THREADS;

	// Size of per-block smem
	// Padded by overhanging radius
	size_t SMEM = (THREADS + r * 2) * sizeof(int);

	auto start = std::chrono::high_resolution_clock::now();
	convolution_1d << < GRID, THREADS, SMEM >> > (device_array, device_result, n);
	auto end = std::chrono::high_resolution_clock::now();
	auto result = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	printf("GPU took %d microseconds \n", result);

	cudaMemcpy(host_result, device_result, bytes, cudaMemcpyDeviceToHost);

	start = std::chrono::high_resolution_clock::now();
	verify_result(host_array, host_mask, host_result, n);
	end = std::chrono::high_resolution_clock::now();
	result = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	printf("CPU took %d microseconds \n", result);

	printf("Completed\n");

	delete[] host_array;
	delete[] host_result;
	delete[] host_mask;
	cudaFree(device_result);

	return 0;
}