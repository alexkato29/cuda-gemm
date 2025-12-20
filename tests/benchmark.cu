#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N_SAMPLES 10


void benchmark_size(int N, cublasHandle_t handle) {
	float *h_A, *h_B, *h_C;
	float alpha = 1.0f, beta = 0.0f;
}


int main() {
	cublasHandle_t handle;
	cublasCreate(&handle);

	std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};

	for (int N : sizes) {
		benchmark_size(N, handle);
	}

	cublasDestroy(handle);
	return 0;
}
