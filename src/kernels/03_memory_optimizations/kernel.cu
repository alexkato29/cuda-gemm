#include "kernel.cuh"

#define BLOCK_DIM 16
#define TILE_DIM 64
#define COARSE_DIM 4


__global__ void memory_optimize(float* A, float* B, float* C, float alpha, float beta, int N) {
	int thread_tile_row_base = threadIdx.y * COARSE_DIM;
	int thread_tile_col_base = threadIdx.x * COARSE_DIM;

	int block_base_row = blockIdx.y * TILE_DIM;
	int block_base_col = blockIdx.x * TILE_DIM;

	// TODO: Optimize the size here to make shared memory fit more warps occupancy??? Do this last probably...
	__shared__ float a_tile[TILE_DIM][TILE_DIM];
	__shared__ float b_tile[TILE_DIM][TILE_DIM];

	float sums[COARSE_DIM][COARSE_DIM] = {0.0f};
	float reg_a[COARSE_DIM];
	float reg_b[COARSE_DIM];

	for (int tile_idx = 0; tile_idx < (N + TILE_DIM - 1) / TILE_DIM; tile_idx++) {
		for (int row_offset_in_tile = 0; row_offset_in_tile < TILE_DIM; row_offset_in_tile += 8) {
			for (int col_offset_in_tile = 0; col_offset_in_tile < TILE_DIM; col_offset_in_tile += 32) {
				int thread_idx = threadIdx.y * 16 + threadIdx.x;
				int row_within_tile = (thread_idx / 32) + row_offset_in_tile;
				int col_within_tile = (thread_idx % 32) + col_offset_in_tile;

				int a_col = col_within_tile + tile_idx * TILE_DIM;
				int a_row = row_within_tile + block_base_row;

				int b_row = row_within_tile + tile_idx * TILE_DIM;
				int b_col = col_within_tile + block_base_col;

				// TODO: Vectorize these memory accesses. A can probably be vectorized easily, maybe not B?
				// Basically, should be able to read a float4 and immediately copy it to shared memory for x4 throughput?
				if (a_col < N && a_row < N)
					a_tile[row_within_tile][col_within_tile] = A[a_row * N + a_col];
				else
					a_tile[row_within_tile][col_within_tile] = 0.0f;

				if (b_row < N && b_col < N)
					b_tile[row_within_tile][col_within_tile] = B[b_row * N + b_col];
				else
					b_tile[row_within_tile][col_within_tile] = 0.0f;
			}
		}
		__syncthreads();

		// TODO: Vectorize these loads (involves transposing B)
		for (int i = 0; i < TILE_DIM; i++) {
			for (int j = 0; j < COARSE_DIM; j++) {
				reg_a[j] = a_tile[thread_tile_row_base + j][i];
				reg_b[j] = b_tile[i][thread_tile_col_base + j];
			}

			for (int cx = 0; cx < COARSE_DIM; cx++)
				for (int cy = 0; cy < COARSE_DIM; cy++)
					sums[cy][cx] += reg_a[cy] * reg_b[cx];
		}
		__syncthreads();
	}

	// TODO: Vectorize the writes. 
	int row = block_base_row + thread_tile_row_base;
	int col = block_base_col + thread_tile_col_base;
	for (int cx = 0; cx < COARSE_DIM; cx++)
		for (int cy = 0; cy < COARSE_DIM; cy++) {
			int row_within_c = row + cy;
			int col_within_c = col + cx;
			if (row_within_c < N && col_within_c < N) {
				if (beta == 0.0f)
					C[row_within_c * N + col_within_c] = alpha * sums[cy][cx];
				else
					C[row_within_c * N + col_within_c] = alpha * sums[cy][cx] + beta * C[row_within_c * N + col_within_c];
			}
		}
}


void kernel(float* d_A, float* d_B, float* d_C, float alpha, float beta, int N) {
	dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
	dim3 gridDim((N + TILE_DIM - 1) / TILE_DIM,
			(N + TILE_DIM - 1) / TILE_DIM);

	memory_optimize<<<gridDim, blockDim>>>(d_A, d_B, d_C, alpha, beta, N);
}


void cleanup_kernel() {
	return;
}
