#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <chrono>

#define SHMEM_SIZE 16 * 16 * 4

using namespace std;
using namespace std::chrono;

__global__ void matMul(int* a, int* b, int* c, int n, int tile_size)
{
	// 2 static sized pieces of shared memory
	__shared__ int A[SHMEM_SIZE];
	__shared__ int B[SHMEM_SIZE];

	// Shortened names
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Global row and column
	int row = by * tile_size + tx;
	int col = bx * tile_size + ty;

	int temp = 0;
	// sweep tiles over whole matrix
	for (int i = 0; i < (n / tile_size); i++)
	{
		/* 
		Every thread in a threadblock loads 1 element into shared memory
		The element location in shared memory corresponds to the thread's position in the threadblock
		e.g. thread [0, 0] loads for A[0 * tile_size + 0] and B[0 * tile_size + 0]
		
		Indexing for A:
				row * n: Indexes global row for this thread
				i * tile_size: Indexes new set of columsn for each iteration
				tx: Indexes column within that set
		Indexing for B:
				i * tile_size * r: Indexes next set of rows each iteration
				ty * n: Indexes row within that set
				col: indexes global column*/
		A[(ty * tile_size) + tx] = a[row * n + (i * tile_size + tx)];
		B[(ty * tile_size) + tx] = b[(i * tile_size * n + ty * n) + col];

		// Ensure threads threads load data before proceeding
		__syncthreads();

		// Calculate all values for this tile
		for (int j = 0; j < tile_size; j++)
		{
			temp += A[(ty * tile_size) + j] * B[(j * tile_size) + tx];
		}
		// Ensure some threads dont progress and change shared memory values
		__syncthreads();
	}
	c[(row * n) + col] = temp;
}

void init_matrices(int* a, int* b, int n)
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


void verify_result(int* a, int* b, int* c, int n)
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

void transpose(int* a, int* a_t, int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; i < n; j++)
		{
			a_t[j * n + i] = a[i * n + j];
		}
	}
}

int main_tile_mat_mul()
{
	// Matrix of 1024 * 1024
	int n = 1 << 12;
	size_t bytes = n * n * sizeof(int);

	// Host pointers
	int* host_a, * host_b, * host_c;
	host_a = (int*)malloc(bytes);
	host_b = (int*)malloc(bytes);
	host_c = (int*)malloc(bytes);

	// Device pointers
	int* device_a, * device_b, * device_c;
	cudaMalloc(&device_a, bytes);
	cudaMalloc(&device_b, bytes);
	cudaMalloc(&device_c, bytes);

	init_matrices(host_a, host_b, n);

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
	matMul <<<grid, threads>>> (device_a, device_b, device_c, n, BLOCK_SIZE);
	auto end = high_resolution_clock::now();
	printf("GPU finished\n");
	auto duration = duration_cast<microseconds>(end - start);
	printf("GPU took %d microseconds\n", duration);

	cudaMemcpy(host_c, device_c, bytes, cudaMemcpyDeviceToHost);

	printf("CPU starts\n");
	start = high_resolution_clock::now();
	verify_result(host_a, host_b, host_c, n);
	end = high_resolution_clock::now();
	duration = duration_cast<microseconds>(end - start);
	printf("CPU took %d microseconds\n", duration);
	printf("COLMPLETED SUCCESSFULLY\n");

	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);

	free(host_a);
	free(host_b);
	free(host_c);

	return 0;
}