# CUDA-SGEMM

This repo walks through several implementations of ***single precision generalized matrix multiplication***, or just **SGEMM**. Each implementation aims to be increasingly optimized than the last.

All kernels were profiled on a **NVIDIA T4 GPU** via Google Cloud Platform.

## Repo Organization
- `src/kernels/` - Different implementations of SGEMM kernels.
- `tests/` - Validation, benchmarking, and profiling test scripts.
- `include/` - Header files for kernel interface and utilities.

## Running Locally
The `CMakeLists` file allows for easy repo compilation via
```
// Creates the build/ directory and makefile for compilation
mkdir build && cmake -B build -DKERNEL=<target kernel>

// Compiles the benchmark, validation, and profiling scripts
cd build && make 
```
#### Options for Target Kernels
Currently Supported: `CUBLAS`, `NAIVE`

The test scripts allow you to run the kernels for yourself on a NVIDIA GPU.
- `./benchmark` — Produces average runtime information for a kernel on several different sizes of matrices.
- `./validate` — Confirms that a kernel output matches cuBLAS output closely (excluding floating point rounding errors).
- `./profile` — Runs the kernel one time on an arbitrary sized, randomly generated matrix. Ideal for profiling with `NSight Systems` or `NSight Compute` (which automatically runs kernels multiple times).

## Optimizations
Each iteration has a `README.md` containing thorough explanations of the optimization. Additionally, they will guide you through how you might have deduced the need for such an optimization yourself via profiling.
