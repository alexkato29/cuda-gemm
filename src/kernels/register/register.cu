#include "kernel.cuh"

#define BLOCK_DIM 16
#define TILE_DIM 64
#define COARSE_DIM 4

__global__ void register_tiled(float* A, float* B, float* C, float alpha, float beta, int N) {
	__shared__ float a_tile[TILE_DIM][TILE_DIM];
	__shared__ float b_tile[TILE_DIM][TILE_DIM];

	float sums[COARSE_DIM][COARSE_DIM] = {0.0f};
	float reg_a[COARSE_DIM];
	float reg_b[COARSE_DIM];

	for (int i = 0; i < (N + TILE_DIM - 1) / TILE_DIM; i++) {
		for (int row_offset = 0; row_offset < TILE_DIM; row_offset += BLOCK_DIM) {
			for (int col_offset = 0; col_offset < TILE_DIM; col_offset += BLOCK_DIM) {
				int a_row = row_offset + threadIdx.y + blockIdx.y * TILE_DIM;
				int a_col = i * TILE_DIM + threadIdx.x + col_offset;

				int b_row = i * TILE_DIM + threadIdx.y + row_offset;
				int b_col = col_offset + threadIdx.x + blockIdx.x * TILE_DIM;

				if (a_col < N && a_row < N)
					a_tile[threadIdx.y + row_offset][threadIdx.x + col_offset] = A[a_row * N + a_col];
				else
					a_tile[threadIdx.y + row_offset][threadIdx.x + col_offset] = 0.0f;

				if (b_row < N && b_col < N)
					b_tile[threadIdx.y + row_offset][threadIdx.x + col_offset] = B[b_row * N + b_col];
				else
					b_tile[threadIdx.y + row_offset][threadIdx.x + col_offset] = 0.0f;
			}
		}
		__syncthreads();

		for (int j = 0; j < TILE_DIM; j++) {
			for (int c = 0; c < COARSE_DIM; c++) {
				reg_a[c] = a_tile[threadIdx.y * COARSE_DIM + c][j];
				reg_b[c] = b_tile[j][threadIdx.x * COARSE_DIM + c];
			}

			for (int cx = 0; cx < COARSE_DIM; cx++)
				for (int cy = 0; cy < COARSE_DIM; cy++)
					sums[cy][cx] += reg_a[cy] * reg_b[cx];
		}
		__syncthreads();
	}

	int row = blockIdx.y * TILE_DIM + threadIdx.y * COARSE_DIM;
	int col = blockIdx.x * TILE_DIM + threadIdx.x * COARSE_DIM;
	for (int cx = 0; cx < COARSE_DIM; cx++)
		for (int cy = 0; cy < COARSE_DIM; cy++) {
			if (row + cy < N && col + cx < N) {
				if (beta == 0.0f)
					C[(row + cy) * N + col + cx] = alpha * sums[cy][cx];
				else
					C[(row + cy) * N + col + cx] = alpha * sums[cy][cx] + beta * C[(row + cy) * N + col + cx];
			}
		}
}


void kernel(float* d_A, float* d_B, float* d_C, float alpha, float beta, int N) {
	dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
	dim3 gridDim((N + TILE_DIM - 1) / TILE_DIM,
			(N + TILE_DIM - 1) / TILE_DIM);

	register_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, alpha, beta, N);
}


void cleanup_kernel() {
	return;
}
