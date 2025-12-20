cmake_minimum_required(VERSION 3.18)
project(cuda-gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_library(utils 
    src/utils/generate_matrix.cu
)
target_include_directories(utils PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(benchmark tests/benchmark.cu)
target_link_libraries(benchmark PRIVATE utils)
