#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

 #include "utils/generate_matrix.cuh"

#define N_SAMPLES 10


static inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

float benchmark_kernel(int N, float alpha, float beta) {
	const int SIZE = N * N * sizeof(float);

	float *h_A = nullptr, *h_B = nullptr, *h_C = nullptr;
	checkCuda(cudaMallocHost(&h_A, SIZE));
	checkCuda(cudaMallocHost(&h_B, SIZE));
	checkCuda(cudaMallocHost(&h_C, SIZE));

	generate_matrix(h_A, SIZE);
	generate_matrix(h_B, SIZE);
	generate_matrix(h_C, SIZE);

	float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
	checkCuda(cudaMalloc(&d_A, SIZE));
    checkCuda(cudaMalloc(&d_B, SIZE));
    checkCuda(cudaMalloc(&d_C, SIZE));

    checkCuda(cudaMemcpy(d_A, h_A, SIZE, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B, SIZE, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_C, h_C, SIZE, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    // TODO: Run the kernel once to warm the cache

    checkCuda(cudaEventRecord(start));

    for (int i = 0; i < N_SAMPLES; ++i) {
    	// TODO: Actually run the kernel
        continue;
    }

    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop));
    ms /= N_SAMPLES;

    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));

    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C));

    checkCuda(cudaFreeHost(h_A));
    checkCuda(cudaFreeHost(h_B));
    checkCuda(cudaFreeHost(h_C));

	return ms;
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

	return 0;
}
