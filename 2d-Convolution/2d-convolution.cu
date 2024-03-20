#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <chrono>

#define MASK_DIM 13

// How much mask will hang over matrix
#define MASK_OFFSET (MASK_DIM / 2)

__constant__ int mask[MASK_DIM * MASK_DIM];

__global__ void convolution_2d(int* matrix, int* result, int n)
{
	// 2d thread positions
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Start idx
	int start_r = row - MASK_OFFSET;
	int start_c = col - MASK_OFFSET;

	int temp = 0;

	for (int i = 0; i < MASK_DIM; i++)
	{
		for (int j = 0; j < MASK_DIM; j++)
		{
			if (start_r + i >= 0 && start_r + i < n && start_c + j >= 0 && start_c + j < n)
			{
				temp += matrix[(start_r + i) * n + (start_c + j)] * mask[i * MASK_DIM + j];
			}
		}
	}
	result[row * n + col] = temp;
}

void verify_result(int* m, int* mask, int* result, int n)
{
	int temp;
	int offset_r;
	int offset_c;

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			temp = 0;
			for (int k = 0; k < MASK_DIM; k++)
			{
				offset_r = i - MASK_OFFSET + k;
				for (int l = 0; l < MASK_DIM; l++)
				{
					offset_c = j - MASK_OFFSET + l;
					if (offset_r >= 0 && offset_r < n && offset_c >= 0 && offset_c < n)
					{
						temp += m[offset_r * n + offset_c] * mask[k * MASK_DIM + l];
					}
				}
			}
			assert(result[i * n + j] == temp);
		}
	}
}

int main()
{
	// 2^10 * 2^10 matrix
	int n = 1 << 12;
	size_t bytes_n = n * n * sizeof(int);

	// Allocate matrix
	int* matrix = new int[n * n];
	int* result = new int[n * n];
	for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) matrix[i * n + j] = rand() % 100;

	// Allocate and init mask
	int* host_mask = new int[MASK_DIM * MASK_DIM];
	size_t bytes_m = MASK_DIM * MASK_DIM * sizeof(int);
	for (int i = 0; i < MASK_DIM; i++) for (int j = 0; j < MASK_DIM; j++) host_mask[i * MASK_DIM + j] = rand() % 100;

	int* device_matrix, * device_result;
	cudaMalloc(&device_matrix, bytes_n);
	cudaMalloc(&device_result, bytes_n);

	cudaMemcpy(device_matrix, matrix, bytes_n, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(mask, host_mask, bytes_m);

	// 16 * 16 blocks
	const int THREADS = 16;
	const int GRID = (n + THREADS - 1) / THREADS;

	dim3 block_dim(THREADS, THREADS);
	dim3 grid_dim(GRID, GRID);

	auto start = std::chrono::high_resolution_clock::now();
	convolution_2d << < grid_dim, block_dim >> > (device_matrix, device_result, n);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	printf("GPU took %d microseconds\n", duration);

	cudaMemcpy(result, device_result, bytes_n, cudaMemcpyDeviceToHost);

	start = std::chrono::high_resolution_clock::now();
	verify_result(matrix, host_mask, result, n);
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	printf("CPU took %d microseconds\n", duration);

	printf("Completed\n");

	return 0;
}
