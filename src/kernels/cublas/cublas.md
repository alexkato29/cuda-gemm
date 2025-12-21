# Baseline: cuBLAS

### Strategy
**cuBLAS** is the NVIDIA, closed-source imeplementation of GEMM. It is highly optimized and used as a baseline for this repository. Unfortunately, information on the full implementation is not known.

### Benchmark Results
```
Average Runtime per Matrix Size:
128x128 Matrix: 0.007502 ms
256x256 Matrix: 0.025431 ms
512x512 Matrix: 0.058717 ms
1024x1024 Matrix: 0.398195 ms
2048x2048 Matrix: 2.827056 ms
4096x4096 Matrix: 19.131596 ms
```