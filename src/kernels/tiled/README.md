# Iteration 1: Tiled Matrix Multiplication

### Strategy
Just as the naive implementation, we still are using each thread to compute one output element via a dot product. But, the memory access pattern changes. Rather than fetching a row and a column from global memory for each output element, we can take advantage of repeated access.

Consider output elements $C_{0,0}$ and $C_{0, 1}$. To compute either element require fetching the first row of $A$. Rather than doing this twice independently, we can utilize *shared memory* to fetch this row a single time.

We divide our output matrix $C$ into `16x16` (we could have chosen a different size, this was arbitrary) tiles. For each tile, we fetch the corresponding 16 rows of `A` and 16 columns of `B` one time and store them in shared memory. 

After this cooperative load, we may efficiently reuse the rows and columns from shared memory, which is substantially faster to read from. It also has a separate memory instruction queue from local/global memory, removing the bottleneck of the naive kernel.

### Benchmark Results
```
Average Runtime per Matrix Size:
128x128 Matrix: 0.015109 ms
256x256 Matrix: 0.074956 ms
512x512 Matrix: 0.515013 ms
1024x1024 Matrix: 3.760020 ms
2048x2048 Matrix: 24.659019 ms
4096x4096 Matrix: 200.523956 ms
```
#### Speedup Factors (on 512+ Matrices)
cuBLAS: 0.1862x
Naive:  1.7578x

### What's Good?
- Tiling dramatically reduces the amount of global memory reads (1/16th as many loads issued in our code, theoretically).
- Pressure on the local/global memory instruction queue is no longer a primary bottleneck.
- We have no bank conflicts in our shared memory access. 
    - `a_tile` accesses reference unique bank IDs altogether per warp.
    - `b_tile` accesses do request only 16 unique bank IDs per 32 threads. But, they request the same address, so the result can be broadcasted!
- The compiler is *already vectorizing reads to `a_tile`!* See the bonus observations section.

### What's Bad?
- We call shared memory so aggressively that it cannot serve reads as fast as they're coming in.

### In-Depth Findings
**TLDR;** we hammer shared memory with reads so frequently that the memory I/O instruction queue (responsible for handling shared memory read instructions) is constantly full. Warps are stalled waiting to schedule their load instructions.

1. We observe that warps are frequently stalled.
      - 76.25% of clock cycles have no eligible warps ready to issue an instruction (Scheduler Statistics).
      - Warps are issuing 0.24 instructions per cycle, very far from the 1.0 theoretical max (Scheduler Statistics).
1. Warps are stalled waiting to issue their memory operations.
      - Warps are spending 17.6 cycles in between instructions, on average, stalled waiting for the memory input/output (MIO) instruction queue to not be full (Warp State Statistics).
1. We are hammering the MIO insturction queue via shared memory accesses.
      - Shared memory reads issue their instructions to MIO instruction queue.
      - The tiled dot product for loop executes 2 reads to shared memory per iteration per thread.
      - We don't issue any special math instructions of dynamic branches.
      - *This is the main bottleneck.*

### In-Depth Fix Ideas
#### Use fewer but wider memory loads.
We currently issue one load instruction per float we want from shared memory. When retrieving a row or column of a matrix, ideally we can just get multiple values from that row or column under one instruction simultaneously. The load store units (LSUs) that actually fetch the shared memory *support this*. They have far more bandwidth than 4 bytes per instruction. 

*What if we read multiple values per instruction over fewer total instructions?*
**Note:** see the *Bonus Observations* section at the bottom of this `README.md`, but the compiler is smart enough to already do this (at least in part).

#### Read from shared memory less frequently.
Shared memory is faster than global memory, but it still takes a non-negligible amount of time. And, we still must repeatedly read the same rows and columns over and over again from shared. Ideally, we could read the shared memory values fewer times yet do more work per read.

*What if we have each thread compute multiple output elements from single shared memory reads?*

### Profiling Results
```
tiled(float *, float *, float *, float, float, int) (128, 128, 1)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 7.5
Section: GPU Speed Of Light Throughput
----------------------- ------------- ------------
Metric Name               Metric Unit Metric Value
----------------------- ------------- ------------
DRAM Frequency          cycle/nsecond         5.00
SM Frequency            cycle/usecond       584.88
Elapsed Cycles                  cycle     27087563
Memory Throughput                   %        74.42
DRAM Throughput                     %        29.27
Duration                      msecond        46.31
L1/TEX Cache Throughput             %        95.38
L2 Cache Throughput                 %        10.67
SM Active Cycles                cycle  27059076.98
Compute (SM) Throughput             %        74.42
----------------------- ------------- ------------

INF   Compute and Memory are well-balanced: To reduce runtime, both computation and memory traffic must be reduced.
    Check both the Compute Workload Analysis and Memory Workload Analysis sections.

Section: GPU Speed Of Light Roofline Chart
INF   The ratio of peak float (fp32) to double (fp64) performance on this device is 32:1. The kernel achieved 12%
    of this device's fp32 peak performance and 0% of its fp64 peak performance. See the Kernel Profiling Guide
    (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline) for more details on roofline
    analysis.

Section: Compute Workload Analysis
-------------------- ----------- ------------
Metric Name          Metric Unit Metric Value
-------------------- ----------- ------------
Executed Ipc Active   inst/cycle         0.95
Executed Ipc Elapsed  inst/cycle         0.95
Issue Slots Busy               %        23.74
Issued Ipc Active     inst/cycle         0.95
SM Busy                        %        26.92
-------------------- ----------- ------------

WRN   All compute pipelines are under-utilized. Either this kernel is very small or it doesn't issue enough warps
    per scheduler. Check the Launch Statistics and Scheduler Statistics sections for further details.

Section: Memory Workload Analysis
----------------- ------------ ------------
Metric Name        Metric Unit Metric Value
----------------- ------------ ------------
Memory Throughput Gbyte/second        93.57
Mem Busy                     %        47.69
Max Bandwidth                %        74.42
L1/TEX Hit Rate              %         0.46
L2 Hit Rate                  %        49.39
Mem Pipes Busy               %        74.42
----------------- ------------ ------------

Section: Memory Workload Analysis Tables
OPT   Estimated Speedup: 5.314%
    The memory access pattern for loads from L1TEX to L2 is not optimal. The granularity of an L1TEX request to
    L2 is a 128 byte cache line. That is 4 consecutive 32-byte sectors per L2 request. However, this kernel only
    accesses an average of 2.0 sectors out of the possible 4 sectors per cache line. Check the Source Counters
    section for uncoalesced loads and try to minimize how many cache lines need to be accessed per memory
    request.
----- --------------------------------------------------------------------------------------------------------------
OPT   Estimated Speedup: 0.02082%
    The memory access pattern for stores from L1TEX to L2 is not optimal. The granularity of an L1TEX request to
    L2 is a 128 byte cache line. That is 4 consecutive 32-byte sectors per L2 request. However, this kernel only
    accesses an average of 2.0 sectors out of the possible 4 sectors per cache line. Check the Source Counters
    section for uncoalesced stores and try to minimize how many cache lines need to be accessed per memory
    request.

Section: Scheduler Statistics
---------------------------- ----------- ------------
Metric Name                  Metric Unit Metric Value
---------------------------- ----------- ------------
One or More Eligible                   %        23.75
Issued Warp Per Scheduler                        0.24
No Eligible                            %        76.25
Active Warps Per Scheduler          warp         7.97
Eligible Warps Per Scheduler        warp         0.74
---------------------------- ----------- ------------

OPT   Estimated Speedup: 76.25%
    Every scheduler is capable of issuing one instruction per cycle, but for this kernel each scheduler only
    issues an instruction every 4.2 cycles. This might leave hardware resources underutilized and may lead to
    less optimal performance. Out of the maximum of 8 warps per scheduler, this kernel allocates an average of
    7.97 active warps per scheduler, but only an average of 0.74 warps were eligible per cycle. Eligible warps
    are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible
    warp results in no instruction being issued and the issue slot remains unused. To increase the number of
    eligible warps, avoid possible load imbalances due to highly different execution durations per warp.
    Reducing stalls indicated on the Warp State Statistics and Source Counters sections can help, too.

Section: Warp State Statistics
---------------------------------------- ----------- ------------
Metric Name                              Metric Unit Metric Value
---------------------------------------- ----------- ------------
Warp Cycles Per Issued Instruction             cycle        33.56
Warp Cycles Per Executed Instruction           cycle        33.56
Avg. Active Threads Per Warp                                   32
Avg. Not Predicated Off Threads Per Warp                    31.99
---------------------------------------- ----------- ------------

OPT   Estimated Speedup: 51.86%
    On average, each warp of this kernel spends 17.4 cycles being stalled waiting for the MIO (memory
    input/output) instruction queue to be not full. This stall reason is high in cases of extreme utilization of
    the MIO pipelines, which include special math instructions, dynamic branches, as well as shared memory
    instructions. When caused by shared memory accesses, trying to use fewer but wider loads can reduce pipeline
    pressure. This stall type represents about 51.9% of the total average of 33.6 cycles between issuing two
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
Avg. Executed Instructions Per Scheduler        inst   6424985.60
Executed Instructions                           inst   1027997696
Avg. Issued Instructions Per Scheduler          inst   6425031.10
Issued Instructions                             inst   1028004976
---------------------------------------- ----------- ------------

Section: Launch Statistics
-------------------------------- --------------- ---------------
Metric Name                          Metric Unit    Metric Value
-------------------------------- --------------- ---------------
Block Size                                                   256
Function Cache Configuration                     CachePreferNone
Grid Size                                                  16384
Registers Per Thread             register/thread              39
Shared Memory Configuration Size           Kbyte           32.77
Driver Shared Memory Per Block        byte/block               0
Dynamic Shared Memory Per Block       byte/block               0
Static Shared Memory Per Block       Kbyte/block            2.05
Threads                                   thread         4194304
Waves Per SM                                              102.40
-------------------------------- --------------- ---------------

Section: Occupancy
------------------------------- ----------- ------------
Metric Name                     Metric Unit Metric Value
------------------------------- ----------- ------------
Block Limit SM                        block           16
Block Limit Registers                 block            6
Block Limit Shared Mem                block           16
Block Limit Warps                     block            4
Theoretical Active Warps per SM        warp           32
Theoretical Occupancy                     %          100
Achieved Occupancy                        %        99.62
Achieved Active Warps Per SM           warp        31.88
------------------------------- ----------- ------------

INF   This kernel's theoretical occupancy is not impacted by any block limit.

Section: Source Counters
------------------------- ----------- ------------
Metric Name               Metric Unit Metric Value
------------------------- ----------- ------------
Branch Instructions Ratio           %         0.02
Branch Instructions              inst     17170432
Branch Efficiency                   %          100
Avg. Divergent Branches                          0
------------------------- ----------- ------------
```

### Bonus Observations

#### Two `__syncthreads()` Calls
We synchronize threads within the block twice. The first time is straightforward. We must make sure shared memory is completely updated before we begin computing dot products. The second synchronization is necessary to prevent certain warps from racing ahead and beginning to update shared memory *again* before all threads have finished using the current shared memory values.

#### Compiler is Already Vectorizing `a_tile` Loads
If we compile the `./profile` binary and disasemble: `cuobjdump -sass ./profile > tiled_dump.txt` we see two particularly interesting block of instructions:
```
LDS.U R16, [R24] ;                             /* 0x0000000018107984 */
LDS.U.128 R4, [R23] ;                          /* 0x0000000017047984 */
LDS.U R18, [R24+0x40] ;                        /* 0x0000400018127984 */
LDS.U R28, [R24+0x80] ;                        /* 0x00008000181c7984 */
LDS.U R29, [R24+0xc0] ;                        /* 0x0000c000181d7984 */
...
FFMA R4, R16, R4, R19 ;                        /* 0x0000000410047223 */
FFMA R5, R18, R5, R4 ;                         /* 0x0000000512057223 */
FFMA R6, R28, R6, R5 ;                         /* 0x000000061c067223 */
FFMA R7, R29, R7, R6 ;                         /* 0x000000071d077223 */
```
I've removed a few in between lines, but this section is reading memory from `a_tile` (register `R23`) and `b_tile` (register `R24`). When we read from `a_tile`, observe we use the instruction `LDS.U.128`. This is reading 128 bits from memory, or 16 bytes, or *4 floats* simultaneously. It's already vectorized and using a wide load with one instruction! The compiler is smart.

#### DRAM Throughput Increased
We observe that DRAM throughput increases from the naive kernel, despite making 1/16th as many global memory calls. Seems counterintuitive, but this can be explained by the L1/TEX cache hit rate. While for naive it was nearly 90%, it has dropped to <1% in the new tiled kernel! We issue less loads from global memory, but now they almost always miss in L1 and end up in L2 and (half the time) all the way in DRAM.

This has to do with both temporal locality. Our previous kernel would have a thread read a row from global memory, then almost immediately request that row again in another thread. Rows and columns could be prefetched and stored on the L1 cache for fast access. Now, shared memory hides this reuse significantly and somewhat assumes the role of the L1 cache (albeit explicit and more efficiently).
