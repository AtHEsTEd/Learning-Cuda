// Cuda libaries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Normal c++ libaries
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

constexpr auto VECTOR_POWER = 16;

using namespace std;

// CUDA kernel for vector addition
__global__ void vectorAdd(int* a, int* b, int* c, int n)
{
	// Calculate thread id
	// Only x vals are accessed as it is 1d
	int id = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Boundary guard
	if (id < n)
	{
		c[id] = a[id] + b[id];
	}
}


// Initialise vector size n with values between 0-99
void matrix_init(int* a, int n)
{
	for (int i = 0; i < n; i++)
	{
		a[i] = rand() % 100;
	}
}

void error_check(int* a, int* b, int* c, int n)
{
	for (int i = 0; i < n; i++)
	{
		assert(c[i] == a[i] + b[i]);
	}
}


int main_vector_add() 
{
	// Vector size of 2^16 - 65536 elements
	// << operator is bit shifting
	int n = 1 << VECTOR_POWER;

	// Host (CPU) vector pointers
	int* host_a, * host_b, * host_c;
	// Device (GPU) vector pointers
	int* device_a, * device_b, * device_c;

	// Memory size in bytes
	size_t bytes = sizeof(int) * n;

	// Allocate cpu memory
	host_a = (int*)malloc(bytes);
	host_b = (int*)malloc(bytes);
	host_c = (int*)malloc(bytes);

	// Allocate gpu memory
	// & before a variable/pointer accesses its address
	cudaMalloc(&device_a, bytes);
	cudaMalloc(&device_b, bytes);
	cudaMalloc(&device_c, bytes);

	// Initialise vectors a and b randomly
	matrix_init(host_a, n);
	matrix_init(host_b, n);

	/* Copy data from cpu to gpu
	 First parameter: variable to copy data into
	 Second parameter: variable to copy data from
	 Third parameter: amount of bytes to copy
	 Fourth parameter: what direction to copy data by */
	cudaMemcpy(device_a, host_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, host_b, bytes, cudaMemcpyHostToDevice);

	// Choose size of threadblock (1d as it's a vector)
	const int NUM_THREADS = 256;

	// Grid size
	const int NUM_BLOCKS = (int)ceil(n / NUM_THREADS);

	// Launch kernel
	vectorAdd <<<NUM_BLOCKS, NUM_THREADS>>> (device_a, device_b, device_c, n);

	// Copy sum from gpu to cpu
	cudaMemcpy(host_c, device_c, bytes, cudaMemcpyDeviceToHost);

	// Error check
	error_check(host_a, host_b, host_c, n);
	printf("COMPLETED SUCCESSFULLY\n");
	return 0;
}