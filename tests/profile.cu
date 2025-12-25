#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "utils/generate_matrix.cuh"
#include "kernel.cuh"


static inline void checkCuda(cudaError_t e) {
	if (e != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(e));
		std::exit(1);
	}
}

static inline void checkCublas(cublasStatus_t status) {
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("cuBLAS error: %d\n", status);
		std::exit(1);
	}
}


int main(int argc, char** argv) {
	int N = 128;
	if (argc > 1) {
		N = std::atoi(argv[1])
	}
	printf("Profiling matrix size: %dx%d\n", N, N);

	const int SIZE = N * N;
	const int BYTES = SIZE * sizeof(float);
	float alpha = 1.0f, beta = 1.0f;

	float *h_A = nullptr, *h_B = nullptr, *h_C = nullptr;
	cudaMallocHost(&h_A, BYTES);
	cudaMallocHost(&h_B, BYTES);
	cudaMallocHost(&h_C, BYTES);
	checkCuda(cudaGetLastError());

	generate_matrix(h_A, SIZE);
	generate_matrix(h_B, SIZE);
	generate_matrix(h_C, SIZE);

	float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
	cudaMalloc(&d_A, BYTES);
	cudaMalloc(&d_B, BYTES);
	cudaMalloc(&d_C, BYTES);
	checkCuda(cudaGetLastError());

	cudaMemcpy(d_A, h_A, BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, BYTES, cudaMemcpyHostToDevice);
	checkCuda(cudaGetLastError());

	kernel(d_A, d_B, d_C, alpha, beta, N);
	checkCuda(cudaDeviceSynchronize());

	cleanup_kernel();

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);

	return 0;
}
