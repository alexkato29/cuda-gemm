#include "kernel.cuh"

#define TILE_DIM 16

__global__ void tiled(float* A, float* B, float* C, float alpha, float beta, int N) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float a_tile[TILE_DIM][TILE_DIM];
	__shared__ float b_tile[TILE_DIM][TILE_DIM];

	float sum = 0.0f;
	for (int i = 0; i < (N + TILE_DIM - 1) / TILE_DIM; i++) {
		int a_col = i * TILE_DIM + threadIdx.x;
		int b_row = i * TILE_DIM + threadIdx.y;

		if (a_col < N && row < N)
			a_tile[threadIdx.y][threadIdx.x] = A[row * N + a_col];
		else
			a_tile[threadIdx.y][threadIdx.x] = 0.0f;

		if (b_row < N && col < N)
			b_tile[threadIdx.y][threadIdx.x] = B[b_row * N + col];
		else
			b_tile[threadIdx.y][threadIdx.x] = 0.0f;

		__syncthreads();

		for (int k = 0; k < TILE_DIM; k++)
			sum += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];

		__syncthreads();
	}

	if (row < N && col < N)
		d_C[row * N + col] = d_C[row * N + col] * beta + sum * alpha;
}


void kernel(float* d_A, float* d_B, float* d_C, float alpha, float beta, int N) {
	dim3 blockDim(TILE_DIM, TILE_DIM);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
			(N + blockDim.y - 1) / blockDim.y);

	tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, alpha, beta, N);
}
