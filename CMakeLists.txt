cmake_minimum_required(VERSION 3.18)
project(cuda-gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(KERNEL "cublas" CACHE STRING "Select which kernel implementation to benchmark")
set_property(CACHE KERNEL PROPERTY STRINGS cublas)

set(KERNEL_FILE "${PROJECT_SOURCE_DIR}/src/kernels/${KERNEL}.cu")
if(NOT EXISTS ${KERNEL_FILE})
    message(FATAL_ERROR "Kernel '${KERNEL}' not found at ${KERNEL_FILE}")
endif()

message(STATUS "Building with kernel: ${KERNEL}")

add_library(utils
    src/utils/generate_matrix.cu
)
target_include_directories(utils PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(benchmark
    tests/benchmark.cu
    ${KERNEL_FILE}
)
target_link_libraries(benchmark PRIVATE utils)
target_include_directories(benchmark PRIVATE ${PROJECT_SOURCE_DIR}/include)

if(KERNEL STREQUAL "cublas")
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(benchmark PRIVATE CUDA::cublas)
endif()
