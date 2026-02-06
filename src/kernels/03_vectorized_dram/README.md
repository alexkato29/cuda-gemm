# Iteration 3: Optimized Memory Layout, Stores, & Loads

### Strategy
Not much has changed in our strategy, but how we store and load to memory is more efficient.
1. Wherever possible, we load `float4` values instead of `float`. This allows us to read multiple floats per instruction.
1. We linearized threads to avoid bank conflicts. In the previous iteration, we were slow due to a "2.1 way bank conflict". That bank conflict is avoided via writing contiguous values.
1. As a bonus, we made our matmul kernel actually flexible to matrix sizes. Note that this is still actually broken, but I am more focused on the performance than thinking about how to deal with remainder rows/cols at this point.

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
- We are, once again, way less memory bound at 64% of peak FP32 performance.
- Linearized global memory loads eliminate the annoying 2.1 way bank conflict from last iteration.
- Memory reads and writes are now mostly vectorized, massively relieving DRAM throughput.

### What's Bad?
- We now see heavy bank conflicts on shared memory stores.
    - Stores are reported as having conflicts by the profiler, but in reality it *should* be conflict free. See bonus observations.

### In-Depth Findings

### In-Depth Fix Ideas

### Profiling Results
```
Section: GPU Speed Of Light Throughput
----------------------- ------------- ------------
Metric Name               Metric Unit Metric Value
----------------------- ------------- ------------
DRAM Frequency          cycle/nsecond         5.00
SM Frequency            cycle/usecond       585.65
Elapsed Cycles                  cycle      5225782
Memory Throughput                   %        44.75
DRAM Throughput                     %         6.75
Duration                      msecond         8.92
L1/TEX Cache Throughput             %        89.51
L2 Cache Throughput                 %         7.37
SM Active Cycles                cycle   4765467.88
Compute (SM) Throughput             %        64.78
----------------------- ------------- ------------

OPT   Compute is more heavily utilized than Memory: Look at the Compute Workload Analysis section to see what the
    compute pipelines are spending their time doing. Also, consider whether any computation is redundant and
    could be reduced or moved to look-up tables.

Section: GPU Speed Of Light Roofline Chart
INF   The ratio of peak float (fp32) to double (fp64) performance on this device is 32:1. The kernel achieved 64%
    of this device's fp32 peak performance and 0% of its fp64 peak performance. See the Kernel Profiling Guide
    (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline) for more details on roofline
    analysis.

Section: Compute Workload Analysis
-------------------- ----------- ------------
Metric Name          Metric Unit Metric Value
-------------------- ----------- ------------
Executed Ipc Active   inst/cycle         1.60
Executed Ipc Elapsed  inst/cycle         1.46
Issue Slots Busy               %        39.92
Issued Ipc Active     inst/cycle         1.60
SM Busy                        %        71.04
-------------------- ----------- ------------

WRN   FMA is the highest-utilized pipeline (71.0%) based on active cycles, taking into account the rates of its
    different instructions. It executes 32-bit floating point (FADD, FMUL, FMAD, ...) and integer (IMUL, IMAD)
    operations. The pipeline is well-utilized, but might become a bottleneck if more work is added. Based on the
    number of executed instructions, the highest utilized pipeline (71.0%) is FMA. It executes 32-bit floating
    point (FADD, FMUL, FMAD, ...) and integer (IMUL, IMAD) operations. Comparing the two, the overall pipeline
    utilization appears to be caused by frequent, low-latency instructions. See the Kernel Profiling Guide
    (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder) or hover over the
    pipeline name to understand the workloads handled by each pipeline. The Instruction Statistics section shows
    the mix of executed instructions in this kernel. Check the Warp State Statistics section for which reasons
    cause warps to stall.

Section: Memory Workload Analysis
----------------- ------------ ------------
Metric Name        Metric Unit Metric Value
----------------- ------------ ------------
Memory Throughput Gbyte/second        21.62
Mem Busy                     %        44.75
Max Bandwidth                %        26.15
L1/TEX Hit Rate              %         8.33
L2 Hit Rate                  %        75.29
Mem Pipes Busy               %        18.12
----------------- ------------ ------------

Section: Memory Workload Analysis Tables
OPT   Estimated Speedup: 0.0166%
    The memory access pattern for global loads in L1TEX might not be optimal. On average, this kernel accesses
    16.0 bytes per thread per memory request; but the address pattern, possibly caused by the stride between
    threads, results in 16.5 sectors per request, or 16.5*32 = 527.5 bytes of cache data transfers per request.
    The optimal thread address pattern for 16.0 byte accesses would result in 16.0*32 = 512.0 bytes of cache
    data transfers per request, to maximize L1TEX cache performance. Check the Source Counters section for
    uncoalesced global loads.
----- --------------------------------------------------------------------------------------------------------------
OPT   Estimated Speedup: 0.2822%
    The memory access pattern for global stores in L1TEX might not be optimal. On average, this kernel accesses
    16.0 bytes per thread per memory request; but the address pattern, possibly caused by the stride between
    threads, results in 32.0 sectors per request, or 32.0*32 = 1024.0 bytes of cache data transfers per request.
    The optimal thread address pattern for 16.0 byte accesses would result in 16.0*32 = 512.0 bytes of cache
    data transfers per request, to maximize L1TEX cache performance. Check the Source Counters section for
    uncoalesced global stores.
----- --------------------------------------------------------------------------------------------------------------
OPT   Estimated Speedup: 36.5%
    The memory access pattern for shared loads might not be optimal and causes on average a 5.0 - way bank
    conflict across all 16777216 shared load requests.This results in 33584000 bank conflicts,  which represent
    40.02% of the overall 83915648 wavefronts for shared loads. Check the Source Counters section for
    uncoalesced shared loads.
----- --------------------------------------------------------------------------------------------------------------
OPT   Estimated Speedup: 9.958%
    The memory access pattern for shared stores might not be optimal and causes on average a 4.5 - way bank
    conflict across all 1048576 shared store requests.This results in 514169 bank conflicts,  which represent
    10.92% of the overall 4708473 wavefronts for shared stores. Check the Source Counters section for
    uncoalesced shared stores.

Section: Scheduler Statistics
---------------------------- ----------- ------------
Metric Name                  Metric Unit Metric Value
---------------------------- ----------- ------------
One or More Eligible                   %        39.94
Issued Warp Per Scheduler                        0.40
No Eligible                            %        60.06
Active Warps Per Scheduler          warp         3.76
Eligible Warps Per Scheduler        warp         0.86
---------------------------- ----------- ------------

OPT   Estimated Speedup: 60.06%
    Every scheduler is capable of issuing one instruction per cycle, but for this kernel each scheduler only
    issues an instruction every 2.5 cycles. This might leave hardware resources underutilized and may lead to
    less optimal performance. Out of the maximum of 8 warps per scheduler, this kernel allocates an average of
    3.76 active warps per scheduler, but only an average of 0.86 warps were eligible per cycle. Eligible warps
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
Warp Cycles Per Issued Instruction             cycle         9.42
Warp Cycles Per Executed Instruction           cycle         9.42
Avg. Active Threads Per Warp                                   32
Avg. Not Predicated Off Threads Per Warp                    31.96
---------------------------------------- ----------- ------------

Section: Instruction Statistics
---------------------------------------- ----------- ------------
Metric Name                              Metric Unit Metric Value
---------------------------------------- ----------- ------------
Avg. Executed Instructions Per Scheduler        inst   1902553.60
Executed Instructions                           inst    304408576
Avg. Issued Instructions Per Scheduler          inst   1902573.40
Issued Instructions                             inst    304411744
---------------------------------------- ----------- ------------

Section: Launch Statistics
-------------------------------- --------------- ---------------
Metric Name                          Metric Unit    Metric Value
-------------------------------- --------------- ---------------
Block Size                                                   256
Function Cache Configuration                     CachePreferNone
Grid Size                                                    256
Registers Per Thread             register/thread             128
Shared Memory Configuration Size           Kbyte           65.54
Driver Shared Memory Per Block        byte/block               0
Dynamic Shared Memory Per Block       byte/block               0
Static Shared Memory Per Block       Kbyte/block           32.77
Threads                                   thread           65536
Waves Per SM                                                3.20
-------------------------------- --------------- ---------------

Section: Occupancy
------------------------------- ----------- ------------
Metric Name                     Metric Unit Metric Value
------------------------------- ----------- ------------
Block Limit SM                        block           16
Block Limit Registers                 block            2
Block Limit Shared Mem                block            2
Block Limit Warps                     block            4
Theoretical Active Warps per SM        warp           16
Theoretical Occupancy                     %           50
Achieved Occupancy                        %        46.94
Achieved Active Warps Per SM           warp        15.02
------------------------------- ----------- ------------

OPT   This kernel's theoretical occupancy (50.0%) is limited by the number of required registers. This kernel's
    theoretical occupancy (50.0%) is limited by the required amount of shared memory. See the CUDA Best
    Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more
    details on optimizing occupancy.

Section: Source Counters
------------------------- ----------- ------------
Metric Name               Metric Unit Metric Value
------------------------- ----------- ------------
Branch Instructions Ratio           %         0.01
Branch Instructions              inst      1576960
Branch Efficiency                   %          100
Avg. Divergent Branches                          0
------------------------- ----------- ------------

OPT   Estimated Speedup: 0%
    This kernel has uncoalesced global accesses resulting in a total of 1048576 excessive sectors (6% of the
    total 18874368 sectors). Check the L2 Theoretical Sectors Global Excessive table for the primary source
    locations. The CUDA Programming Guide
    (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses) had additional
    information on reducing uncoalesced device memory accesses.
----- --------------------------------------------------------------------------------------------------------------
OPT   Estimated Speedup: 34.74%
    This kernel has uncoalesced shared accesses resulting in a total of 33554432 excessive wavefronts (38% of the
    total 88080384 wavefronts). Check the L1 Wavefronts Shared Excessive table for the primary source locations.
    The CUDA Best Practices Guide
     (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c
    -aa) has an example on optimizing shared memory accesses.
```

### Bonus Observations
