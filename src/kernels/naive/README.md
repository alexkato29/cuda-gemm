# Iteration 0: Naive Dot Product

### Strategy
We let each thread compute one element of the output matrix `C = A @ B`. Each thread with index $(x, y)$ will:
1. Load row $x$ of `A` and column $y$ of `B`.
2. Perform the dot product of these two rows.
3. Write the output to `C` in DRAM.

### Benchmark Results
```
Average Runtime per Matrix Size:
128x128 Matrix: 0.027169 ms
256x256 Matrix: 0.157083 ms
512x512 Matrix: 1.137391 ms
1024x1024 Matrix: 6.414844 ms
2048x2048 Matrix: 38.661644 ms
4096x4096 Matrix: 310.596100 ms
```
**cuBLAS Factor (512+ Matrices): 9.4594x**

### What's Good?
- Memory access in `B` is coalesced.

### What's Bad?
- Warp geometry (`16x2`) is suboptimal for the problem.
	- Changing to `32x1` gives a 13.7% performance boost.
- Expensive cache misses lead to LG throttling.
- Memory reads are highly redundant and inefficient.
