# Iteration 2: Register Tiling

### Strategy
Last time, we ran into pressure on the MIO instruction queue due to overwhelming shared memory load instructions. In this iteration, we reduce the number of calls to shared memory by having each thread compute more outputs per memory read.

Rather than one thread computing one element of $C$, we now designate *each thread to compute a `4x4` tile of outputs*. For example, the thread previously responsible for $C_{i, j}$ will now compute $C_{i:i+3, j:j+3}$. This is accomplished via:

1. Loading a tile of A and tile of B to shared memory.
2. Loading 4 rows of `a_tile` and 4 columns of `b_tile` into *registers*.
3. Performing 16 dot products to compute 16 partial outputs.
4. Re-performing these loads and computations for all required tiles of `A` and `B`.
5. Writing the `4x4` output to `C`.

Loading from registers takes 0 cycles and adds no pressure to any queues. While before it took 2 reads per 1 output, we now perform 8 reads per 16 outputs.

***Note:** `4x4` was a semi-arbitrary tile size. It was the fastest of `4x4` and `8x8`, but other sizes I did not test/reason about.*

### Benchmark Results
```
Average Runtime per Matrix Size:
128x128 Matrix: 0.029148 ms
256x256 Matrix: 0.049698 ms
512x512 Matrix: 0.230535 ms
1024x1024 Matrix: 1.039047 ms
2048x2048 Matrix: 7.867392 ms
4096x4096 Matrix: 56.125244 ms
```

#### Speedup Factors (on 512+ Matrices)
```
cuBLAS: 0.5747x
Naive:  5.3889x
Prev:   3.1400x
```

### What's Good?
- MIO instruction queue pressure is significantly reduced.
- The kernel is far less memory bound, reaching 47% of peak FP32 performance.

### What's Bad?
- There are bank conflicts when writing to the shared memory.
- Our Streaming Multiprocessors (SMs) are only 50% occupied due to shared memory resource requirements.
- The MIO instruction queue is continuing to see *some* pressure (albeit 1/3rd as much as the previous kernel).

### In-Depth Findings
**TLDR;** poor memory access and storage patterns are causing use to lose a lot of efficiency when reading and writing to *both* global and shared memory.

1. Warps continue to be stalled waiting on the MIO instruction queue to be not full.
	- It is much improved from the tiled kernel, but warps are still waiting 6.0 cycles for the queue on average (Warp State Statistics).
1. Writes take extra time due to bank conflicts.
      - The average store to shared memory experiences a 2.1 way bank conflict (Memory Workload Analysis).
      - Bank conflicts on stores increase the store time by 51.28% (Memory Workload Analysis).
1. Reads from shared memory are not coalesced.
      - We do 7.006% extra work per shared memory read due to excessive wavefronts from uncoalesced access, on average (Source Counters).
1. No reads are vectorized (SASS).
      - The compiler is not vectorizing any memory reads or writes.
1. We achieve poor warp occupancy.
      - We use 32 kb of shared memory per block.
      - Due to the 64 kb shared memory constraint, we can only fit 2 blocks per SM (Occupancy).

### In-Depth Fix Ideas
#### Use fewer but wider memory loads.
None of our global memory nor shared memory reads and writes are vectorized. We are using 4x as much memory bandiwdth as we could be while performing memory operations.

*What if we vectorized memory reads to read multiple elements from global memory at once?*

***Note:** the compiler might be doing this on my machine, but I used [GodBolt](https://godbolt.org) to check.*

#### Eliminate bank conflicts.
When writing to `a_tile` and `b_tile`, we experience bank conflicts due to writing two rows simultaneously. Padding won't trivially fix this either, as adding padding will increase the size of our (already large) shared memory bank and would require significant padding to offset. 

*What if we optimized shared memory writes to write to 32 separate banks?*

#### Suboptimal tile sizing.
In the current configuration, tiles are so large that we only get half warp occupancy. There is probably a better balance to be struck between maximizing the amount of shared memory per block while also maximizing SM occupancy.

*What if we sized tiles methodically to maximize the amount of shared memory and warps per tile?*

### Profiling Results
```
register_tiled(float *, float *, float *, float, float, int) (32, 32, 1)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 7.5
Section: GPU Speed Of Light Throughput
----------------------- ------------- ------------
Metric Name               Metric Unit Metric Value
----------------------- ------------- ------------
DRAM Frequency          cycle/nsecond         5.00
SM Frequency            cycle/usecond       584.94
Elapsed Cycles                  cycle      7187283
Memory Throughput                   %        48.97
DRAM Throughput                     %        21.48
Duration                      msecond        12.29
L1/TEX Cache Throughput             %        97.94
L2 Cache Throughput                 %        10.81
SM Active Cycles                cycle   7049744.38
Compute (SM) Throughput             %        47.15
----------------------- ------------- ------------

OPT   This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance
      of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate
      latency issues. Look at Scheduler Statistics and Warp State Statistics for potential reasons.

Section: GPU Speed Of Light Roofline Chart
INF   The ratio of peak float (fp32) to double (fp64) performance on this device is 32:1. The kernel achieved 47%
      of this device's fp32 peak performance and 0% of its fp64 peak performance. See the Kernel Profiling Guide
      (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline) for more details on roofline
      analysis.

Section: Compute Workload Analysis
-------------------- ----------- ------------
Metric Name          Metric Unit Metric Value
-------------------- ----------- ------------
Executed Ipc Active   inst/cycle         1.28
Executed Ipc Elapsed  inst/cycle         1.26
Issue Slots Busy               %        32.07
Issued Ipc Active     inst/cycle         1.28
SM Busy                        %        48.07
-------------------- ----------- ------------

INF   FMA is the highest-utilized pipeline (48.1%) based on active cycles, taking into account the rates of its
      different instructions. It executes 32-bit floating point (FADD, FMUL, FMAD, ...) and integer (IMUL, IMAD)
      operations. It is well-utilized, but should not be a bottleneck.

Section: Memory Workload Analysis
----------------- ------------ ------------
Metric Name        Metric Unit Metric Value
----------------- ------------ ------------
Memory Throughput Gbyte/second        68.68
Mem Busy                     %        48.97
Max Bandwidth                %        38.14
L1/TEX Hit Rate              %         9.20
L2 Hit Rate                  %        60.46
Mem Pipes Busy               %        38.14
----------------- ------------ ------------

Section: Memory Workload Analysis Tables
OPT   Estimated Speedup: 0.2382%
      The memory access pattern for global stores in L1TEX might not be optimal. On average, this kernel accesses
      11.3 bytes per thread per memory request; but the address pattern, possibly caused by the stride between
      threads, results in 16.0 sectors per request, or 16.0*32 = 512.0 bytes of cache data transfers per request.
      The optimal thread address pattern for 11.3 byte accesses would result in 11.3*32 = 363.2 bytes of cache
      data transfers per request, to maximize L1TEX cache performance. Check the Source Counters section for
      uncoalesced global stores.
----- --------------------------------------------------------------------------------------------------------------
OPT   Estimated Speedup: 4.574%
      The memory access pattern for loads from L1TEX to L2 is not optimal. The granularity of an L1TEX request to
      L2 is a 128 byte cache line. That is 4 consecutive 32-byte sectors per L2 request. However, this kernel only
      accesses an average of 2.2 sectors out of the possible 4 sectors per cache line. Check the Source Counters
      section for uncoalesced loads and try to minimize how many cache lines need to be accessed per memory
      request.
----- --------------------------------------------------------------------------------------------------------------
OPT   Estimated Speedup: 51.28%
      The memory access pattern for shared stores might not be optimal and causes on average a 2.1 - way bank
      conflict across all 8388608 shared store requests.This results in 9189141 bank conflicts,  which represent
      52.28% of the overall 17577749 wavefronts for shared stores. Check the Source Counters section for
      uncoalesced shared stores.

Section: Scheduler Statistics
---------------------------- ----------- ------------
Metric Name                  Metric Unit Metric Value
---------------------------- ----------- ------------
One or More Eligible                   %        32.05
Issued Warp Per Scheduler                        0.32
No Eligible                            %        67.95
Active Warps Per Scheduler          warp         3.93
Eligible Warps Per Scheduler        warp         0.62
---------------------------- ----------- ------------

OPT   Estimated Speedup: 67.95%
      Every scheduler is capable of issuing one instruction per cycle, but for this kernel each scheduler only
      issues an instruction every 3.1 cycles. This might leave hardware resources underutilized and may lead to
      less optimal performance. Out of the maximum of 8 warps per scheduler, this kernel allocates an average of
      3.93 active warps per scheduler, but only an average of 0.62 warps were eligible per cycle. Eligible warps
      are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible
      warp results in no instruction being issued and the issue slot remains unused. To increase the number of
      eligible warps, avoid possible load imbalances due to highly different execution durations per warp.
      Reducing stalls indicated on the Warp State Statistics and Source Counters sections can help, too.
----- --------------------------------------------------------------------------------------------------------------
OPT   Estimated Speedup: 50%
      The 4.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the
      hardware maximum of 8. Use the Occupancy section to identify what limits this kernel's theoretical occupancy.

Section: Warp State Statistics
---------------------------------------- ----------- ------------
Metric Name                              Metric Unit Metric Value
---------------------------------------- ----------- ------------
Warp Cycles Per Issued Instruction             cycle        12.27
Warp Cycles Per Executed Instruction           cycle        12.27
Avg. Active Threads Per Warp                                   32
Avg. Not Predicated Off Threads Per Warp                    31.59
---------------------------------------- ----------- ------------

OPT   Estimated Speedup: 49.14%
      On average, each warp of this kernel spends 6.0 cycles being stalled waiting for the MIO (memory
      input/output) instruction queue to be not full. This stall reason is high in cases of extreme utilization of
      the MIO pipelines, which include special math instructions, dynamic branches, as well as shared memory
      instructions. When caused by shared memory accesses, trying to use fewer but wider loads can reduce pipeline
      pressure. This stall type represents about 49.1% of the total average of 12.3 cycles between issuing two
      instructions.
----- --------------------------------------------------------------------------------------------------------------
INF   Check the Warp Stall Sampling (All Samples) table for the top stall locations in your source based on
      sampling data. The Kernel Profiling Guide
      (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference) provides more details
      on each stall reason.

Section: Instruction Statistics
---------------------------------------- ----------- ------------
Metric Name                              Metric Unit Metric Value
---------------------------------------- ----------- ------------
Avg. Executed Instructions Per Scheduler        inst   2260531.20
Executed Instructions                           inst    361684992
Avg. Issued Instructions Per Scheduler          inst   2260548.70
Issued Instructions                             inst    361687792
---------------------------------------- ----------- ------------

Section: Launch Statistics
-------------------------------- --------------- ---------------
Metric Name                          Metric Unit    Metric Value
-------------------------------- --------------- ---------------
Block Size                                                   256
Function Cache Configuration                     CachePreferNone
Grid Size                                                   1024
Registers Per Thread             register/thread              64
Shared Memory Configuration Size           Kbyte           65.54
Driver Shared Memory Per Block        byte/block               0
Dynamic Shared Memory Per Block       byte/block               0
Static Shared Memory Per Block       Kbyte/block           32.77
Threads                                   thread          262144
Waves Per SM                                               12.80
-------------------------------- --------------- ---------------

Section: Occupancy
------------------------------- ----------- ------------
Metric Name                     Metric Unit Metric Value
------------------------------- ----------- ------------
Block Limit SM                        block           16
Block Limit Registers                 block            4
Block Limit Shared Mem                block            2
Block Limit Warps                     block            4
Theoretical Active Warps per SM        warp           16
Theoretical Occupancy                     %           50
Achieved Occupancy                        %        49.21
Achieved Active Warps Per SM           warp        15.75
------------------------------- ----------- ------------

OPT   This kernel's theoretical occupancy (50.0%) is limited by the required amount of shared memory. See the CUDA
      Best Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for
      more details on optimizing occupancy.

Section: Source Counters
------------------------- ----------- ------------
Metric Name               Metric Unit Metric Value
------------------------- ----------- ------------
Branch Instructions Ratio           %         0.01
Branch Instructions              inst      2637824
Branch Efficiency                   %          100
Avg. Divergent Branches                          0
------------------------- ----------- ------------

OPT   Estimated Speedup: 0%
      This kernel has uncoalesced global accesses resulting in a total of 3145728 excessive sectors (8% of the
      total 37748736 sectors). Check the L2 Theoretical Sectors Global Excessive table for the primary source
      locations. The CUDA Programming Guide
      (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses) had additional
      information on reducing uncoalesced device memory accesses.
----- --------------------------------------------------------------------------------------------------------------
OPT   Estimated Speedup: 7.006%
      This kernel has uncoalesced shared accesses resulting in a total of 8388608 excessive wavefronts (7% of the
      total 117440512 wavefronts). Check the L1 Wavefronts Shared Excessive table for the primary source
      locations. The CUDA Best Practices Guide
       (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c
      -aa) has an example on optimizing shared memory accesses.
```

### Bonus Observations

#### Global Memory Reads are NOT Vectorized.
Using [GodBolt](https://godbolt.org), we can look at the SASS for this code. It is lengthy (many loops unrolled), but the key takeaway is that no reads are vectorized. It is possible that in my own code the compiler is vectorizing certain loads (it did in the previous iteration), but better to be explicit.

#### Shared Memory Cooperative Load Changes
Block sizes are now smaller than the tile sizing. To completely fill shared memory, we must have each thread load multiple element from global memory to fill the tile.
