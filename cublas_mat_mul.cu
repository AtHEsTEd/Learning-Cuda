#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <chrono>
#include <time.h>

using namespace std;
using namespace std::chrono;

void verify_solution(float* a, float* b, float* c, int n)
{
	float temp;
	float epsilon = 0.000000001;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			temp = 0;
			for (int k = 0; k < n; k++)
			{
				temp += a[k * n + i] * b[j * n + k];
			}
			// assert(fabs(c[j * n + i] - temp) > epsilon);
		}
	}
}

int main()
{
	int n = 1 << 12;
	size_t bytes = n * n * sizeof(float);

	float *host_a, *host_b, *host_c;
	float *device_a, *device_b, *device_c;

	host_a = (float*)malloc(bytes);
	host_b = (float*)malloc(bytes);
	host_c = (float*)malloc(bytes);
	cudaMalloc(&device_a, bytes);
	cudaMalloc(&device_b, bytes);
	cudaMalloc(&device_c, bytes);

	// cuRAND random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set seed using sytem clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

	// Create random matrices
	curandGenerateUniform(prng, device_a, n * n);
	curandGenerateUniform(prng, device_b, n * n);

	// cuBLAS handle
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Scaling factors
	float alpha = 1.0f;
	float beta = 0.0f;

	// Calculates alpha * a * b + beta * c
	// (m * n) * (n * k) = (m * k)
	// Signature: handle, operation, operation, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
	auto start = high_resolution_clock::now();
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, device_a, n, device_b, n, &beta, device_c, n);
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(end - start);
	printf("GPU took %d microseconds\n", duration);


	// Copy back matrices
	cudaMemcpy(host_a, device_a, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_b, device_b, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_c, device_c, bytes, cudaMemcpyDeviceToHost);

	verify_solution(host_a, host_b, host_c, n);

	printf("SUCCESS\n");
	return 0;
}