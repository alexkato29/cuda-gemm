# Iteration 3: Optimized Memory Layout, Stores, & Loads

### Strategy
Not much has changed in our strategy, but how we store and load to memory is more efficient.
1. Wherever possible, we load `float4` values instead of `float`. This allows us to read multiple floats per instruction.
1. We linearized threads to avoid bank conflicts. In the previous iteration, we were slow due to a "2.1 way bank conflict". This is avoided via writing contiguous values.
1. As a bonus, we made our matmul kernel actually flexible to matrix sizes.

### Benchmark Results
```
Average Runtime per Matrix Size:
128x128 Matrix: 0.063428 ms
256x256 Matrix: 0.118919 ms
512x512 Matrix: 0.229237 ms
1024x1024 Matrix: 0.906307 ms
2048x2048 Matrix: 5.959341 ms
4096x4096 Matrix: 34.65185 ms
```

#### Speedup Factors (on 512+ Matrices)
```
cuBLAS: 0.7346x
Naive:  6.8726x
Prev:   1.2730x
```

### What's Good?

### What's Bad?

### In-Depth Findings

### In-Depth Fix Ideas

### Profiling Results
```

```

### Bonus Observations
