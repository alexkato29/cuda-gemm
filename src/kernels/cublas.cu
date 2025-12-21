#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

#include "kernel.cuh"

static inline void checkCublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS error: %d\n", status);
        std::exit(1);
    }
}

void kernel(float* d_A, float* d_B, float* d_C, float alpha, float beta, int N) {
    static cublasHandle_t handle = nullptr;

    if (handle == nullptr) {
        checkCublas(cublasCreate(&handle));
    }

    checkCublas(cublasSgemm(
        handle, 
        CUBLAS_OP_N, 
        CUBLAS_OP_N,
        N, N, N,
        &alpha,
        d_A, N,
        d_B, N,
        &beta,
        d_C, N
    ));
}
