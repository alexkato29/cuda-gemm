#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdlib>


void benchmark_size(int N, cublasHandle_t handle) {
	cout << "Hello world"
}


int main() {
	cublasHandle_t handle;
    cublasCreate(&handle);

    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};

    for (int N : sizes) {
        benchmark_size(N, handle);
    }

    cublasDestroy(handle);
    return 0;
}