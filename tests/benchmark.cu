#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

#include "utils/generate_matrix.cuh"
#include "kernel.cuh"

#define N_SAMPLES 30


static inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

float benchmark_kernel(int N, float alpha, float beta) {
	const int SIZE = N * N;
	const int BYTES = SIZE * sizeof(float);

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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    kernel(d_A, d_B, d_C, alpha, beta, N);
    checkCuda(cudaDeviceSynchronize());

    cudaEventRecord(start);

    for (int i = 0; i < N_SAMPLES; ++i) {
    	kernel(d_A, d_B, d_C, alpha, beta, N);
    }

    cudaEventRecord(stop);
    checkCuda(cudaEventSynchronize(stop));

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

	return ms / N_SAMPLES;
}


int main() {
	srand(256);  // Used for matrix generation

	std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};
	std::vector<float> kernel_results;
	kernel_results.reserve(sizes.size());

	for (int N : sizes) {
		float runtime = benchmark_kernel(N, 1.0f, 0.0f);
		kernel_results.push_back(runtime);
	}

	for (int i = 0; i < kernel_results.size(); i++) {
		printf("%dx%d Matrix: %f ms (average)\n", sizes[i], sizes[i], kernel_results[i]);
	}

	return 0;
}
