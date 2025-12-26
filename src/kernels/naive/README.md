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
	- Changing to `32x1` gave a 13.7% performance boost in a test.
- Expensive cache misses lead to LG throttling.
- Memory reads are highly redundant and inefficient.

### In-Depth Findings
**TLDR;** expensive memory reads lead to LG throttling. Warps are stalled waiting for the local/global memory instruction queue to flush so they may issue subsequent instructions.

1. We observe that warps are frequently stalled.
	- 83.09% of clock cycles have no eligible warps ready to issue an instruction (Scheduler Statistics).
    - Each scheduler is issuing 0.17 instructions per cycle, on average. Very far from the 1.0 theoretical max (Scheduler Statistics).
1. Warps are stalled waiting to issue their memory operations.
	- Warps spend 37.9 out of their 47.16 cycles between instructions, on average, stalled waiting for the local/global memory instruction queue to free up (Warp State Statistics).
    - The local/global memory instruction queue has finite space, and as long as it's full warps can't even begin other independent operations. They must first get the load instruction in flight.
    - *This is the main bottleneck.*
1. Why is the local/global memory instruction queue saturated?
	- It must be either (1) we are reading too often or (2) reads are too expensive.
1. Cache miss rate is not negligble.
	- Cache misses occur 12.7% of the time (Memory Workload Analysis).
1. Memory access patterns are inefficient.
	- Our kernel is using only 1.3 of the 4 sectors per returned cache line (Memory Workload Analysis).
    - We waste the majority of data per cache line and end up requiring more cache lines to obtain all data.
    - With a non-negligible cache miss rate and many cache lines requested, the probability of at least one miss per load instruction is dramatically amplified.
1. We experience many cache misses and queue pressure.
	- Fetching values from DRAM and L2 take considerably longer, potentially 200-400+ cycles.
    - While that is happening, the memory instruction is stuck on the instruction queue.

### In-Depth Fix Ideas
#### Make reads less expensive.
If we can reduce cache misses, we can flush the queue faster, on average. We can do so by fetching less cache lines. A very simple trick to accomplish this will be to **change the geometry of our thread blocks**.

The kernel uses blocks of size `16x16`. The for loop line:
```
sum += d_A[row * N + i] * d_B[i * N + col];
```
*is coalesced* in its access to `B`, but still is only accessing 16 floats * 4 bytes per float = 64 bytes per cache line, while 128 are available! Also, there are two rows per warp and thus we fetch two cache lines for the (strided) accesses to `A`.

By shaping our blocks to `32x8`, we can use *all 128 bytes per cache line* of `B` (32 floats × 4 bytes per float) and only request one cache line of `A` (only one `row` value per warp)!

#### Push less to the local/global memory instruction queue.
We reference global memory many times per thread block. But, we can leverage shared memory to alleviate pressure on the L1 cache/DRAM and subsequently the memory instruction queue. Shared memory is on chip and accessed very efficiently. It does *NOT* rely on local/global memory instruction queue, and its contents are guaranteed to be there at runtime. Plus, shared memory enables very efficient data reuse. This is especially valuable for our situation, where row and column reads are highly redundant (each row of `A` and col of `B` is read `N` times for an `NxN` matrix).

### Profiling Results
```
Section: GPU Speed Of Light Throughput
----------------------- ------------- ------------
Metric Name               Metric Unit Metric Value
----------------------- ------------- ------------
DRAM Frequency          cycle/nsecond         5.00
SM Frequency            cycle/usecond       600.08
Elapsed Cycles                  cycle     43701903
Memory Throughput                   %        61.48
DRAM Throughput                     %        18.80
Duration                      msecond        72.83
L1/TEX Cache Throughput             %        92.24
L2 Cache Throughput                 %         7.01
SM Active Cycles                cycle  43659048.95
Compute (SM) Throughput             %        61.48
----------------------- ------------- ------------

INF   Compute and Memory are well-balanced: To reduce runtime, both computation and memory traffic must be reduced.
      Check both the Compute Workload Analysis and Memory Workload Analysis sections.

Section: GPU Speed Of Light Roofline Chart
INF   The ratio of peak float (fp32) to double (fp64) performance on this device is 32:1. The kernel achieved 8% of
      this device's fp32 peak performance and 0% of its fp64 peak performance. See the Kernel Profiling Guide
      (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline) for more details on roofline
      analysis.

Section: Compute Workload Analysis
-------------------- ----------- ------------
Metric Name          Metric Unit Metric Value
-------------------- ----------- ------------
Executed Ipc Active   inst/cycle         0.68
Executed Ipc Elapsed  inst/cycle         0.68
Issue Slots Busy               %        16.90
Issued Ipc Active     inst/cycle         0.68
SM Busy                        %        20.52
-------------------- ----------- ------------

WRN   All compute pipelines are under-utilized. Either this kernel is very small or it doesn't issue enough warps
      per scheduler. Check the Launch Statistics and Scheduler Statistics sections for further details.

Section: Memory Workload Analysis
----------------- ------------ ------------
Metric Name        Metric Unit Metric Value
----------------- ------------ ------------
Memory Throughput Gbyte/second        60.13
Mem Busy                     %        46.12
Max Bandwidth                %        61.48
L1/TEX Hit Rate              %        87.30
L2 Hit Rate                  %        50.36
Mem Pipes Busy               %        61.48
----------------- ------------ ------------

Section: Memory Workload Analysis Tables
OPT   Estimated Speedup: 4.649%
      The memory access pattern for loads from L1TEX to L2 is not optimal. The granularity of an L1TEX request to
      L2 is a 128 byte cache line. That is 4 consecutive 32-byte sectors per L2 request. However, this kernel only
      accesses an average of 1.3 sectors out of the possible 4 sectors per cache line. Check the Source Counters
      section for uncoalesced loads and try to minimize how many cache lines need to be accessed per memory
      request.
----- --------------------------------------------------------------------------------------------------------------
OPT   Estimated Speedup: 0.01353%
      The memory access pattern for stores from L1TEX to L2 is not optimal. The granularity of an L1TEX request to
      L2 is a 128 byte cache line. That is 4 consecutive 32-byte sectors per L2 request. However, this kernel only
      accesses an average of 2.0 sectors out of the possible 4 sectors per cache line. Check the Source Counters
      section for uncoalesced stores and try to minimize how many cache lines need to be accessed per memory
      request.

Section: Scheduler Statistics
---------------------------- ----------- ------------
Metric Name                  Metric Unit Metric Value
---------------------------- ----------- ------------
One or More Eligible                   %        16.91
Issued Warp Per Scheduler                        0.17
No Eligible                            %        83.09
Active Warps Per Scheduler          warp         7.97
Eligible Warps Per Scheduler        warp         0.70
---------------------------- ----------- ------------

OPT   Estimated Speedup: 83.09%
      Every scheduler is capable of issuing one instruction per cycle, but for this kernel each scheduler only
      issues an instruction every 5.9 cycles. This might leave hardware resources underutilized and may lead to
      less optimal performance. Out of the maximum of 8 warps per scheduler, this kernel allocates an average of
      7.97 active warps per scheduler, but only an average of 0.70 warps were eligible per cycle. Eligible warps
      are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible
      warp results in no instruction being issued and the issue slot remains unused. To increase the number of
      eligible warps, avoid possible load imbalances due to highly different execution durations per warp.
      Reducing stalls indicated on the Warp State Statistics and Source Counters sections can help, too.

Section: Warp State Statistics
---------------------------------------- ----------- ------------
Metric Name                              Metric Unit Metric Value
---------------------------------------- ----------- ------------
Warp Cycles Per Issued Instruction             cycle        47.16
Warp Cycles Per Executed Instruction           cycle        47.16
Avg. Active Threads Per Warp                                   32
Avg. Not Predicated Off Threads Per Warp                    31.98
---------------------------------------- ----------- ------------

OPT   Estimated Speedup: 80.27%
      On average, each warp of this kernel spends 37.9 cycles being stalled waiting for the L1 instruction queue
      for local and global (LG) memory operations to be not full. Typically, this stall occurs only when executing
      local or global memory instructions extremely frequently. Avoid redundant global memory accesses. Try to
      avoid using thread-local memory by checking if dynamically indexed arrays are declared in local scope, of if
      the kernel has excessive register pressure causing by spills. If applicable, consider combining multiple
      lower-width memory operations into fewer wider memory operations and try interleaving memory operations and
      math instructions. This stall type represents about 80.3% of the total average of 47.2 cycles between
      issuing two instructions.
----- --------------------------------------------------------------------------------------------------------------
INF   Check the Warp Stall Sampling (All Samples) table for the top stall locations in your source based on
      sampling data. The Kernel Profiling Guide
      (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference) provides more details
      on each stall reason.

Section: Instruction Statistics
---------------------------------------- ----------- ------------
Metric Name                              Metric Unit Metric Value
---------------------------------------- ----------- ------------
Avg. Executed Instructions Per Scheduler        inst   7378534.40
Executed Instructions                           inst   1180565504
Avg. Issued Instructions Per Scheduler          inst   7378563.59
Issued Instructions                             inst   1180570174
---------------------------------------- ----------- ------------

Section: Launch Statistics
-------------------------------- --------------- ---------------
Metric Name                          Metric Unit    Metric Value
-------------------------------- --------------- ---------------
Block Size                                                   256
Function Cache Configuration                     CachePreferNone
Grid Size                                                  16384
Registers Per Thread             register/thread              49
Shared Memory Configuration Size           Kbyte           32.77
Driver Shared Memory Per Block        byte/block               0
Dynamic Shared Memory Per Block       byte/block               0
Static Shared Memory Per Block        byte/block               0
Threads                                   thread         4194304
Waves Per SM                                              102.40
-------------------------------- --------------- ---------------

Section: Occupancy
------------------------------- ----------- ------------
Metric Name                     Metric Unit Metric Value
------------------------------- ----------- ------------
Block Limit SM                        block           16
Block Limit Registers                 block            4
Block Limit Shared Mem                block           16
Block Limit Warps                     block            4
Theoretical Active Warps per SM        warp           32
Theoretical Occupancy                     %          100
Achieved Occupancy                        %        99.63
Achieved Active Warps Per SM           warp        31.88
------------------------------- ----------- ------------

INF   This kernel's theoretical occupancy is not impacted by any block limit.

Section: Source Counters
------------------------- ----------- ------------
Metric Name               Metric Unit Metric Value
------------------------- ----------- ------------
Branch Instructions Ratio           %         0.02
Branch Instructions              inst     17956864
Branch Efficiency                   %          100
Avg. Divergent Branches                          0
------------------------- ----------- ------------
```

### Bonus Observations

#### SM Frequency != Clock Speed
The T4 supports a max clock speed of ~1.5 GHz. We are using a datacenter GPU, and it is likely not boosted to its theoretical max clock speed. Regardless, boosting to 1.5GHz as we'll see probably won't really make a difference anyway. We aren't compute bound.
