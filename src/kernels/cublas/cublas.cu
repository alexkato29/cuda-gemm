#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

#include "kernel.cuh"

static cublasHandle_t g_handle = nullptr;

static inline void checkCublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS error: %d\n", status);
        std::exit(1);
    }
}


void kernel(float* d_A, float* d_B, float* d_C, float alpha, float beta, int N) {
    if (g_handle == nullptr) {
        checkCublas(cublasCreate(&g_handle));
    }

    checkCublas(cublasSgemm(
    g_handle, 
    CUBLAS_OP_N, 
    CUBLAS_OP_N,
    N, N, N,
    &alpha,
    d_B, N,
    d_A, N,
    &beta,
    d_C, N
    ));
}


void cleanup_kernel() {
    if (g_handle != nullptr) {
        checkCublas(cublasDestroy(g_handle));
        g_handle = nullptr;
    }
}
