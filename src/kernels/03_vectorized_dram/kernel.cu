#include "kernel.cuh"


template <const int SHARED_M, const int SHARED_K, const int SHARED_N, const int OUT_M, const int OUT_N>
__global__ void memory_optimize(int M, int N, int K, float* A, float* B, float* C, float alpha, float beta) {
	// Technically this is fine. I as the programmer control OUT_N.
	static_assert(OUT_N % 4 == 0, "OUT_N must be a multiple of 4 for vectorized B loads and float4 C stores.");

	float sums[OUT_M][OUT_N] = {0.0f};
	float reg_a[OUT_M];
	float reg_b[OUT_N];

	__shared__ float a_tile[SHARED_M][SHARED_K];
	__shared__ float b_tile[SHARED_K][SHARED_N];

	// Determine the base row of A and column of B that this block is responsible for
	int base_a_row = blockIdx.y * SHARED_M;
	int base_b_col = blockIdx.x * SHARED_N;

	// The exact row/col this thread maps to in SMEM
	int thread_row_within_smem = threadIdx.x / (SHARED_M / OUT_M) * OUT_M;    // 2 unique vals per warp
	int thread_col_within_smem = (threadIdx.x % (SHARED_N / OUT_N)) * OUT_N;  // 16 unique vals per warp

	int a_tile_float4s = SHARED_M * SHARED_K / 4;
	int b_tile_float4s = SHARED_K * SHARED_N / 4;

	for (int tile_idx = 0; tile_idx < K; tile_idx += SHARED_K) {
		for (int idx = threadIdx.x; idx < a_tile_float4s; idx += blockDim.x) {
			int a_tile_row = idx / (SHARED_K / 4);
			int a_tile_col = (idx % (SHARED_K / 4)) * 4;
			int a_coord = (base_a_row + a_tile_row) * K + tile_idx + a_tile_col;

			// So, while this store shouldn't have bank conflicts, the profiler says it does?
			// I think it might be a profiler oddity. See readme.
			reinterpret_cast<float4 *>(&a_tile[a_tile_row][a_tile_col])[0] = 
				reinterpret_cast<float4 *>(&A[a_coord])[0];
		}

		for (int idx = threadIdx.x; idx < b_tile_float4s; idx += blockDim.x) {
			int b_tile_row = idx / (SHARED_N / 4);
			int b_tile_col = (idx % (SHARED_N / 4)) * 4;
			int b_coord = (tile_idx + b_tile_row) * N + base_b_col + b_tile_col;

			reinterpret_cast<float4 *>(&b_tile[b_tile_row][b_tile_col])[0] = 
				reinterpret_cast<float4 *>(&B[b_coord])[0];
		}
		__syncthreads();

		#pragma unroll
		for (int k = 0; k < SHARED_K; k++) {
			#pragma unroll
			for (int i = 0; i < OUT_M; i++)
				/* We linearized threads, so they mostly share thread_row_within_smem values.
				This is actually a broadcast in disguise. Not a perfect one, but a
				(32 / (SHARED_M / OUT_M)) way conflict. Far less than a 32 way conflict...*/
				reg_a[i] = a_tile[thread_row_within_smem + i][k];
			#pragma unroll
			for (int i = 0; i < OUT_N; i += 4) {
				/* This is actually a WORSE bank conflict than the a_tile read. thread_col_within_smem
				has 16 unique values per warp, incrementing by OUT_N. Broadcasting makes the overlap 
				efficient, but the OUT_N increment is problematic. We only use 32 / OUT_N unique banks. 
				In this case, a 4.0 way conflict! */
				float4 b_vec = reinterpret_cast<float4*>(&b_tile[k][thread_col_within_smem + i])[0];
				reg_b[i + 0] = b_vec.x;
				reg_b[i + 1] = b_vec.y;
				reg_b[i + 2] = b_vec.z;
				reg_b[i + 3] = b_vec.w;
			}
			#pragma unroll
			for (int cx = 0; cx < OUT_N; cx++)
				#pragma unroll
				for (int cy = 0; cy < OUT_M; cy++)
					sums[cy][cx] += reg_a[cy] * reg_b[cx];
		}
		__syncthreads();
	}	

	int base_out_row = base_a_row + thread_row_within_smem;
	int base_out_col = base_b_col + thread_col_within_smem;
	for (int tile_m = 0; tile_m < OUT_M; tile_m += 1) {
		int row = base_out_row + tile_m;
		for (int tile_n = 0; tile_n < OUT_N; tile_n += 4) {
			int col = base_out_col + tile_n;
			float4 c_vec = reinterpret_cast<float4*>(&C[row * N + col])[0];
			c_vec.x = alpha * sums[tile_m][tile_n + 0] + beta * c_vec.x;
			c_vec.y = alpha * sums[tile_m][tile_n + 1] + beta * c_vec.y;
			c_vec.z = alpha * sums[tile_m][tile_n + 2] + beta * c_vec.z;
			c_vec.w = alpha * sums[tile_m][tile_n + 3] + beta * c_vec.w;
			reinterpret_cast<float4*>(&C[row * N + col])[0] = c_vec;
		}
	}
}


void kernel(float* d_A, float* d_B, float* d_C, float alpha, float beta, int N) {
	const int SM = 128;
	const int SN = 128;
	const int SK = 32;
	const int OM = 8;
	const int ON = 8;

	dim3 blockDim((SM * SN) / (OM * ON));
	dim3 gridDim((N + SN - 1) / SN, (N + SM - 1) / SM);

	memory_optimize<SM, SK, SN, OM, ON><<<gridDim, blockDim>>>(N, N, N, d_A, d_B, d_C, alpha, beta);
}

void cleanup_kernel() { return; }
