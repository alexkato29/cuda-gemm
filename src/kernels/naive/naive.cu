#include "kernel.cuh"


__global__ void naive(float* d_A, float* d_B, float* d_C, float alpha, float beta, int N) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= N || col >= N) {
		return;
	}

	float sum = 0.0f;
	for (int i = 0; i < N; i++) {
		sum += d_A[row * N + i] * d_B[i * N + col];
	}

	d_C[row * N + col] = d_C[row * N + col] * beta + sum * alpha;
}


void kernel(float* d_A, float* d_B, float* d_C, float alpha, float beta, int N) {
	dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                 (N + blockDim.y - 1) / blockDim.y);
    
    naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
}


void cleanup_kernel() {
    return;
}