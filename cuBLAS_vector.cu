#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <chrono>

using namespace std;

void vector_init(float* a, int n)
{
	for (int i = 0; i < n; i++)
	{
		a[i] = (float)(rand() % 100);
	}
}

void verify_result_cublas_vector(float* a, float* b, float* c, float factor, int n)
{
	for (int i = 0; i < n; i++)
	{
		assert(c[i] == factor * a[i] + b[i]);
	}
}

int main_vector_add_cublas()
{
	int n = 1 << 16;
	size_t bytes = n * sizeof(float);

	float *host_a, *host_b, *host_c;
	float *device_a, *device_b;

	host_a = (float*)malloc(bytes);
	host_b = (float*)malloc(bytes);
	host_c = (float*)malloc(bytes);

	cudaMalloc(&device_a, bytes);
	cudaMalloc(&device_b, bytes);

	vector_init(host_a, n);
	vector_init(host_b, n);
	
	// Initialise handle
	cublasHandle_t handle;
	cublasCreate_v2(&handle);

	// Copy vectors to gpu
	cublasSetVector(n, sizeof(float), host_a, 1, device_a, 1);
	cublasSetVector(n, sizeof(float), host_b, 1, device_b, 1);

	// Launch saxpy kernel (single precision a * x + y)
	const float scale = 2.0f;
	cublasSaxpy(handle, n, &scale, device_a, 1, device_b, 1);

	// Copy result back
	cublasGetVector(n, sizeof(float), device_b, 1, host_c, 1);

	verify_result_cublas_vector(host_a, host_b, host_c, scale, n);

	// Garbage collection
	cublasDestroy(handle);
	cudaFree(device_a);
	cudaFree(device_b);
	free(host_a);
	free(host_c);

	return 0;
}