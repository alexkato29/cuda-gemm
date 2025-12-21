#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "utils/generate_matrix.cuh"
#include "kernel.cuh"

#define N 128


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


int main() {
	const int SIZE = N * N;
	const int BYTES = SIZE * sizeof(float);
	float alpha = 1.0f, beta = 1.0f;

	float *h_A = nullptr, *h_B = nullptr, *h_C = nullptr, *h_REF = nullptr;
	cudaMallocHost(&h_A, BYTES);
	cudaMallocHost(&h_B, BYTES);
	cudaMallocHost(&h_C, BYTES);
	cudaMallocHost(&h_REF, BYTES);
    checkCuda(cudaGetLastError());

    generate_matrix(h_A, SIZE);
	generate_matrix(h_B, SIZE);
	generate_matrix(h_C, SIZE);

	for (int i = 0; i < SIZE; i++) {
		h_REF[i] = h_C[i];
	}

	float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_REF = nullptr;
	cudaMalloc(&d_A, BYTES);
    cudaMalloc(&d_B, BYTES);
    cudaMalloc(&d_C, BYTES);
    cudaMalloc(&d_REF, BYTES);
    checkCuda(cudaGetLastError());

    cudaMemcpy(d_A, h_A, BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_REF, h_REF, BYTES, cudaMemcpyHostToDevice);
    checkCuda(cudaGetLastError());

	static cublasHandle_t handle;
	checkCublas(cublasCreate(&handle));
	checkCublas(cublasSgemm(
        handle, 
        CUBLAS_OP_N, 
        CUBLAS_OP_N,
        N, N, N,
        &alpha,
        d_B, N,
        d_A, N,
        &beta,
        d_REF, N
    ));
    checkCublas(cublasDestroy(handle));

	kernel(d_A, d_B, d_C, alpha, beta, N);

	cudaMemcpy(h_REF, d_REF, BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_C, d_C, BYTES, cudaMemcpyDeviceToHost);
	checkCuda(cudaDeviceSynchronize());

	const float TOL = 1e-4f;
	for (int i = 0; i < SIZE; i++) {
		float diff = fabs(h_REF[i] - h_C[i]);
		if (diff > TOL) {
			 printf("Kernel FAILED: mismatch at position %d (expected %f, got %f, diff=%e)\n", i, h_REF[i], h_C[i], diff);
			std::exit(1);
		}
	}
	printf("Kernel VALIDATED: output exactly matches cuBLAS\n");

    cleanup_kernel();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_REF);

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_REF);

	return 0;
}
