#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <chrono>

using namespace std;
using namespace std::chrono;


__global__ void matMul(int* a, int* b, int* c, int n)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int temp_sum = 0;
	if ((row < n) && (col < n))
	{
		// Iterate over row, and down column
		for (int k = 0; k < n; k++)
		{
			temp_sum += a[row * n + k] * b[k * n + col];
		}
		c[row * n + col] = temp_sum;
	}
}


void init_matrices_E(int* a, int* b, int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			a[i * n + j] = rand() % 100;
			b[i * n + j] = rand() % 100;
		}
	}
}


void verify_result_E(int* a, int* b, int* c, int n)
{
	int* verify_c;
	verify_c = (int*)malloc(n * n * sizeof(int));

	for (int i = 0; i < n * n; i++)
	{
		verify_c[i] = 0;
	}


	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			for (int k = 0; k < n; k++)
			{
				verify_c[i * n + j] += a[i * n + k] * b[k * n + j];
			}
		}
	}
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			assert(c[i * n + j] == verify_c[i * n + j]);
		}
	}
}


int main_mat_mul()
{
	// Matrix of 1024 * 1024
	int n = 1 << 12;
	size_t bytes = n * n * sizeof(int);

	// Host pointers
	int *host_a, *host_b, *host_c;
	host_a = (int*)malloc(bytes);
	host_b = (int*)malloc(bytes);
	host_c = (int*)malloc(bytes);

	// Device pointers
	int *device_a, *device_b, *device_c;
	cudaMalloc(&device_a, bytes);
	cudaMalloc(&device_b, bytes);
	cudaMalloc(&device_c, bytes);

	init_matrices_E(host_a, host_b, n);

	// Copy to gpu
	cudaMemcpy(device_a, host_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, host_b, bytes, cudaMemcpyHostToDevice);

	// Threads per block
	int BLOCK_SIZE = 32;

	// Blocks in each dimension
	int GRID_SIZE = (int)ceil(static_cast<double>(n) / BLOCK_SIZE);

	dim3 grid(GRID_SIZE, GRID_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	printf("GPU starts\n");
	auto start = high_resolution_clock::now();
	matMul <<<grid, threads >>> (device_a, device_b, device_c, n);
	auto end = high_resolution_clock::now();
	printf("GPU finished\n");
	auto duration = duration_cast<microseconds>(end - start);
	printf("GPU took %d microseconds\n", duration);

	cudaMemcpy(host_c, device_c, bytes, cudaMemcpyDeviceToHost);

	printf("CPU starts\n");
	start = high_resolution_clock::now();
	verify_result_E(host_a, host_b, host_c, n);
	end = high_resolution_clock::now();
	duration = duration_cast<microseconds>(end - start);
	printf("CPU took %d microseconds\n", duration);
	printf("COLMPLETED SUCCESSFULLY\n");

	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);

	return 0;
}