#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

constexpr auto VECTOR_POWER = 16;

using namespace std;


__global__ void vectorAddUM(int* a, int* b, int* c, int n)
{
	int id = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (id < n)
	{
		c[id] = a[id] + b[id];
	}
}

void init_vector(int* a, int* b, int n)
{
	for (int i = 0; i < n; i++)
	{
		a[i] = rand() % 100;
		b[i] = rand() % 100;
	}
}

void check_answer(int* a, int* b, int* c, int n)
{
	for (int i = 0; i < n; i++)
	{
		assert(c[i] == a[i] + b[i]);
	}
}


int main_vector_add_um()
{
	// Get gpu id
	int id = cudaGetDevice(&id);

	// No. of elements per array
	int n = 1 << VECTOR_POWER;

	size_t bytes = n * sizeof(int);
	int* a, * b, * c;

	// Memory allocation done automatically
	cudaMallocManaged(&a, bytes);
	cudaMallocManaged(&b, bytes);
	cudaMallocManaged(&c, bytes);

	init_vector(a, b, n);
	
	const int BLOCK_SIZE = 256;
	const int GRID_SIZE = (int)ceil(n / static_cast<double>(BLOCK_SIZE));

	// Call CUDA kernal
	// Uncomment these for prefetching params to device
	cudaMemPrefetchAsync(a, bytes, id);
	cudaMemPrefetchAsync(b, bytes, id);
	vectorAddUM << <GRID_SIZE, BLOCK_SIZE >> > (a, b, c, n);

	// Wait for gpu to finish before using values
	cudaDeviceSynchronize();

	// Uncomment for pre-fetching to host
	cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

	check_answer(a, b, c, n);

	printf("COMPLETED SUCCESSFULLY\n");
	return 0;
}