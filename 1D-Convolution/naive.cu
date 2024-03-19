#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <chrono>


__global__ void convolution_1d(int* array, int* mask, int* result, int n, int m)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int r = m / 2;
	int start = tid - r;
	int temp = 0;

	for (int j = 0; j < m; j++)
	{
		if ((start + j >= 0) && (start + j < n))
		{
			temp += array[start + j] * mask[j];
		}
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
	// Result array
	int n = 1 << 22;
	size_t bytes_n = n * sizeof(int);

	// Convolution mask
	int m = 7;
	size_t bytes_m = m * sizeof(int);

	// Allocation and init of host array
	int* host_array = new int[n];
	for (int i = 0; i < n; i++) host_array[i] = rand() % 100;

	// Allocate and init mask
	int* host_mask = new int[m];
	for (int i = 0; i < m; i++) host_mask[i] = rand() % 100;

	// Result
	int* host_result = new int[n];

	int* device_array, * device_mask, * device_result;
	cudaMalloc(&device_array, bytes_n);
	cudaMalloc(&device_mask, bytes_m);
	cudaMalloc(&device_result, bytes_n);

	cudaMemcpy(device_array, host_array, bytes_n, cudaMemcpyHostToDevice);
	cudaMemcpy(device_mask, host_mask, bytes_m, cudaMemcpyHostToDevice);

	const int THREADS = 256;
	const int GRID = (n + THREADS - 1) / THREADS;

	auto start = std::chrono::high_resolution_clock::now();
	convolution_1d << < GRID, THREADS >> > (device_array, device_mask, device_result, n, m);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	printf("GPU took %d microseconds \n", duration);

	cudaMemcpy(host_result, device_result, bytes_n, cudaMemcpyDeviceToHost);

	start = std::chrono::high_resolution_clock::now();
	verify_result(host_array, host_mask, host_result, n, m);
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	printf("CPU took %d microseconds \n", duration);

	printf("Successful\n");

	return 0;
}