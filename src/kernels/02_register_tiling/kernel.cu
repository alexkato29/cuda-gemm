#include "kernel.cuh"

#define BLOCK_DIM 16
#define TILE_DIM 64
#define COARSE_DIM 4


__global__ void register_tiled(float* A, float* B, float* C, float alpha, float beta, int N) {
	// Base coordinates of this particular threads COARSE_DIM x COARSE_DIM output box
	int thread_tile_row_base = threadIdx.y * COARSE_DIM;
	int thread_tile_col_base = threadIdx.x * COARSE_DIM;

	// Base coordinates of this particular threads block
	int block_base_row = blockIdx.y * TILE_DIM;
	int block_base_col = blockIdx.x * TILE_DIM;

	__shared__ float a_tile[TILE_DIM][TILE_DIM];
	__shared__ float b_tile[TILE_DIM][TILE_DIM];

	float sums[COARSE_DIM][COARSE_DIM] = {0.0f};
	float reg_a[COARSE_DIM];
	float reg_b[COARSE_DIM];

	for (int tile_idx = 0; tile_idx < (N + TILE_DIM - 1) / TILE_DIM; tile_idx++) {
		// Because the block dimension is < the tile dimension, each thread must read multiple elements.
		for (int row_offset_in_tile = 0; row_offset_in_tile < TILE_DIM; row_offset_in_tile += BLOCK_DIM) {
			for (int col_offset_in_tile = 0; col_offset_in_tile < TILE_DIM; col_offset_in_tile += BLOCK_DIM) {
				int row_within_tile = threadIdx.y + row_offset_in_tile;
				int col_within_tile = threadIdx.x + col_offset_in_tile;

				// Keep a static row, but move across cols to get ROWS of A
				int a_col = col_within_tile + tile_idx * TILE_DIM;
				int a_row = row_within_tile + block_base_row;

				// Keep a static col, but move across rows to get COLS of B
				int b_row = row_within_tile + tile_idx * TILE_DIM;
				int b_col = col_within_tile + block_base_col;

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

		/*
		This loop computes the dot products. Specifically, it loads the ith value from COARSE_DIM rows of A and
		the ith value from COARSE_DIM cols of B. For i = 0 and these matrices:

		A:						B:						C:
		----------------		----------------		----------------
		|X . . . . . . |		|X X X X . . . |		|X X X X . . . |
		|X . . . . . . |		|. . . . . . . |	 	|X X X X . . . |
		|X . . . . . . |		|. . . . . . . |		|X X X X . . . |
		|X . . . . . . |		|. . . . . . . |		|X X X X . . . |
		----------------		----------------		----------------

		We load in X values and compute the outer products to form a partial region of C.
		Of course, we will need to continue summing values to C for all i.
		*/
		for (int i = 0; i < TILE_DIM; i++) {
			// Load a value from COARSE_DIM rows of A and COARSE_DIM cols of B
			for (int j = 0; j < COARSE_DIM; j++) {
				reg_a[j] = a_tile[thread_tile_row_base + j][i];
				reg_b[j] = b_tile[i][thread_tile_col_base + j];
			}

			// Perform the outer product
			for (int cx = 0; cx < COARSE_DIM; cx++)
				for (int cy = 0; cy < COARSE_DIM; cy++)
					sums[cy][cx] += reg_a[cy] * reg_b[cx];
		}
		__syncthreads();
	}

	// Now we are writing an entire region back to C, not just one value
	int row = block_base_row + thread_tile_row_base;
	int col = block_base_col + thread_tile_col_base;
	for (int cx = 0; cx < COARSE_DIM; cx++)
		for (int cy = 0; cy < COARSE_DIM; cy++) {
			int row_within_c = row + cy;
			int col_within_c = col + cx;
			if (row_within_c < N && col_within_c < N) {
				// Avoid a global memory read if possible
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

	register_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, alpha, beta, N);
}


void cleanup_kernel() {
	return;
}
