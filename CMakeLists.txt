cmake_minimum_required(VERSION 3.18)
project(cuda-gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(KERNEL_PATH "src/kernels/cublas/cublas.cu" CACHE STRING "Path to the kernel implementation file")

if(NOT IS_ABSOLUTE ${KERNEL_PATH})
    set(KERNEL_FILE "${PROJECT_SOURCE_DIR}/${KERNEL_PATH}")
else()
    set(KERNEL_FILE "${KERNEL_PATH}")
endif()

if(NOT EXISTS ${KERNEL_FILE})
    message(FATAL_ERROR "Kernel file not found at ${KERNEL_FILE}")
endif()

message(STATUS "Building with kernel: ${KERNEL_FILE}")

add_library(utils
    src/utils/generate_matrix.cu
)
target_include_directories(utils PUBLIC ${PROJECT_SOURCE_DIR}/include)

find_package(CUDAToolkit REQUIRED)

add_executable(benchmark
    tests/benchmark.cu
    ${KERNEL_FILE}
)
target_link_libraries(benchmark PRIVATE utils CUDA::cublas)
target_include_directories(benchmark PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(validate
    tests/validate.cu
    ${KERNEL_FILE}
)
target_link_libraries(validate PRIVATE utils CUDA::cublas)
target_include_directories(validate PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(profile
    tests/profile.cu
    ${KERNEL_FILE}
)
target_link_libraries(profile PRIVATE utils CUDA::cublas)
target_include_directories(profile PRIVATE ${PROJECT_SOURCE_DIR}/include)
