#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <chrono>

#define size 265

using std::accumulate;
using std::generate;
using std::cout;
using std::vector;
using namespace std;
using namespace std::chrono;

__global__ void sum_reduction(int* vector, int* vector_result)
{

	int id = blockidx.x * blockdim.x + threadidx.x;
	__shared__ int partial_sum[size];

	// load into shared memory
	partial_sum[threadidx.x] = vector[id];
	__syncthreads();

	// lg iterations of block dimension
	for (int i = 1; i < blockdim.x; i *= 2)
	{
		// reduce num of threads by half than previous iteration
		if (threadidx.x % (i * 2) == 0)
		{
			partial_sum[threadidx.x] += partial_sum[threadidx.x + i];
		}
		__syncthreads();
	}
	// only need thread 0 to write the result to vram for each block
	if (threadidx.x == 0)
	{
		vector_result[blockidx.x] = partial_sum[0];
	}
}


void init_vector(int* vector, int n)
{
	for (int i = 0; i < n; i++)
	{
		vector[i] = 1;
	}
}


int main()
{
	// vector size
	int n = 1 << 16;
	size_t bytes = n * sizeof(int);

	// host data
	vector<int> h_v(n);
	vector<int> h_v_r(n);

	// initialize the input data
	generate(begin(h_v), end(h_v), []() { return rand() % 10; });

	// allocate device memory
	int* d_v, * d_v_r;
	cudamalloc(&d_v, bytes);
	cudamalloc(&d_v_r, bytes);

	// copy to device
	cudamemcpy(d_v, h_v.data(), bytes, cudamemcpyhosttodevice);

	// tb size
	const int tb_size = 256;

	// grid size (no padding)
	int grid_size = n / tb_size;

	// call kernels
	auto start = high_resolution_clock::now();
	sum_reduction << <grid_size, tb_size >> > (d_v, d_v_r);
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(end - start);
	printf("gpu took %d microseconds\n", duration);

	sum_reduction << <1, tb_size >> > (d_v_r, d_v_r);

	// copy to host;
	cudamemcpy(h_v_r.data(), d_v_r, bytes, cudamemcpydevicetohost);

	// print the result
	printf("%d\n", h_v_r[0]);
	// assert(h_v_r[0] == accumulate(begin(h_v), end(h_v), 0));

	cout << "completed successfully\n";

	return 0;
}
