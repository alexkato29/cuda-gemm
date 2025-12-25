# Baseline: cuBLAS

### Strategy
**cuBLAS** is the NVIDIA, closed-source imeplementation of GEMM. It is highly optimized and used as a baseline for this repository. Unfortunately, information on the full implementation is not known.

cuBLAS will serve as the baseline throughout this repo. We aim to close the gap between our custom performance and cuBLAS.

### Benchmark Results
```
Average Runtime per Matrix Size:
128x128 Matrix: 0.013248 ms
256x256 Matrix: 0.041482 ms
512x512 Matrix: 0.108889 ms
1024x1024 Matrix: 0.676638 ms
2048x2048 Matrix: 4.908642 ms
4096x4096 Matrix: 30.948292 ms
```