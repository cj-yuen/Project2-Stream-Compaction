CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Christopher Yuen
  * [LinkedIn](https://www.linkedin.com/in/christopher-yuen-16b5a8221/)
* Tested on: Windows 11, 14th Gen Intel(R) Core(TM) i9-14900HX @ 2.22GHz 32GB, RTX 4090 Laptop GPU 16GB (Personal Laptop)

## Stream Compaction & Parallel Scan

Stream compaction removes 0's from an int array while preserving order, and it's the exact operation needed in a path tracer to compact arrays of ray paths after some have terminated. My implementation covers 5 different scan (prefix sum) methods, starting from a serial CPU loop to a work-efficient GPU version with custom tail optimization.

This pipeline has 3 steps:
1) **Map** - each element becomes 1 or 0
2) **Scan** - exclusive prefix sum over the mapped array giving each nonzero element its output index
3) **Scatter** - nonzero elements are written to `odata[indices[i]]`

There are 2 parallel scan variants on GPU:
* **Naive** - `O(n log n)` work, no race condition but heavy
* **Work-Efficient** - `O(n)` work via binary tree up-sweep & down-sweep

## Feature List
* <ins>CPU Scan:</ins> exclusive scan (`scan`), compaction without scan (`compactWithoutScan`), and compaction with scan + scatter (`compactWithScan`) which serves as the correctness baseline for all GPU tests
* <ins>Naive GPU Scan:</ins> ping-pong device buffers across `ilog2ceil(n)` levels
* <ins>Work-Efficient GPU Scan:</ins> uses 1) up-sweep, 2) root-zero, 3) down-sweep over a padded power-of-2 buffer with a separate shift-right kernel to produce the exclusive result
* <ins>Work-Efficient GPU Compaction:</ins> uses `kernMapToBoolean` + work-efficient scan + `kernScatter`
* <ins>Thrust Scan:</ins> wrapper around `thrust::exclusive_scan` on pre-allocated `device_vector`s so only the scan itself is timed
* <ins>Tail Optimization:</ins> collapses the last `~log_2(blockSize)` levels of the up/down-sweeps into single-block kernels, cutting kernel launches roughly in half
* Full support for non-power-of-2 arrays by padding intermediate buffers to the next power of 2

---

## Part 1: CPU Scan & Stream Compaction
Three serial loops:
1) `scan` - the classic single-pass exclusive prefix sum where we write the running total and then add the current element. 
2) `compactWithoutScan` - uses a single write pointer to copy every nonzero element forward in order.
3) `compactWithScan` - does map $\rightarrow$ scan $\rightarrow$ scatter in three passes, matching the GPU pipeline structurally so I can compare its output against `compactWithoutScan`.

## Part 2: Naive GPU Scan
Each level `d` uses an offset of `2^(d-1)` and every element `k >= offset` sums itself with the element `offset` positions to its left for `ilog2ceil(n)` levels. Since threads race when reading and writing the same array, we alternate between two device buffers (`dev_a` $\leftrightarrow$ `dev_b`) each level. A final `kernShiftRight` shifts the inclusive result right by one and inserts 0 at index 0 to get the exclusive scan. This works for both power-of-two and non-power-of-two sizes.

## Part 3: Work-Efficient GPU Scan & Stream Compaction
Based on GPU Gems 3, Chapter 39 - [Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html), operating on a power-of-two padded buffer. 
1) **Up-sweep** - does a parallel reduction up the binary tree
2) **Root Zero** - sets the last element to 0
3) **Down-sweep** - traverses back down, passing each node's value to its left child and setting its right child to the sum

Since no thread writes a location another thread reads in the same level, the scan runs fully in place.

Compaction wraps this: 
1) `kernMapToBoolean` - produces the 0/1 array
2) `cudaMemcpy` - moves it into the padded scan buffer
3) `scanDevice` - exclusive-scans it
4) `kernScatter` - writes `idata[i]` to `odata[indices[i]]` wherever `bools[i] == 1`

To get the total count, I read back just the last scanned value plus the last input element — one int's worth of D2H traffic instead of a full reduction. Also non-power-of-two arrays are handled by rounding up to `n_pow2` and memset-ing the padding to 0.

## Part 4: Using Thrust's Implementation
A thin wrapper around `thrust::exclusive_scan`. I construct the device input and output vectors outside the timer so only the scan call is measured. This excludes the implicit host-to-device copy when building a `device_vector` from a `host_vector` and the final device-to-host `thrust::copy`. I also added `#include <thrust/copy.h>` to pull in `thrust::copy` as it was not running properly otherwise.

## Part 5: Tail Optimization (Extra Credit)
The default work-efficient scan launches `2*log_2(n) + 1` kernels and at the deepest levels `count = n_pow2 >> (d+1)` drops below `blockSize`. Therefore the last few launches are full blocks where nearly every thread is idle. My optimization detects the first level where `count <= blockSize` and collapses all remaining levels of that sweep into a single block kernel that loops over levels with `__syncthreads()` between them. Down-sweep mirrors this. With these optimizations, kernel launches get cut roughly in half.

| n | No tail (ms) | With tail (ms) | Speedup |
|---:|---:|---:|---:|
| 2^10 | 0.650 | 0.275 | 2.36x |
| 2^14 | 0.400 | 0.331 | 1.21x |
| 2^18 | 0.494 | 0.272 | 1.82x |
| 2^22 | 0.959 | 0.615 | 1.56x |
| 2^25 | 6.342 | 6.591 | 0.96x |

The speedup is most clearly seen at lower values of `n`, where the launch overhead has the most impact. At `n = 2^25`, the tail collapses the deepest 20 levels into just 2 kernels which brings the total of 51 launches down to 33. However, at this size, the multi-block portion is doing most of the work so the 2 tail kernels spend longer executing than the individual levels they replace, so the net time is roughly the same. 

---

## Performance Analysis 
### <ins>Block Size Optimization</ins>
Each GPU implementation was tested with the following block sizes `64, 128, 256, 512, 1024` at `n = 2^22` before final data collection, so the tables below compare roughly-optimized implementations rather than unoptimized ones.

| blockSize | Naive (ms) | Work-Efficient (ms) |
|---:|---:|---:|
| 64   | 1.934 | 2.840 |
| 128  | 0.856 | 0.732 |
| 256  | 0.880 | 1.074 |
| 512  | 0.823 | 0.763 |
| 1024 | 1.391 | 0.610 |

<img alt="image" src="https://github.com/user-attachments/assets/c3f8b970-9499-400a-b6dd-f74f79c8962c" />

I chose **blockSize = 512** for both implementations. It's the best Naive time and a reasonable choice for Work-Efficient time. Performance from 128-512 was essentially the same so any of those could work. However, 64 was too small and 1024 hurts the memory-bound naive scan.


### <ins>GPU vs CPU Scan Comparison</ins>
All measurements in **Release x64** on an RTX 4090 Laptop GPU, V-Sync off. Data collected by sweeping the test harness across five sizes (`SIZE = 2^10, 2^14, 2^18, 2^22, 2^25`). Every GPU implementation has its `cudaMalloc`, `cudaMemset`, `cudaMemcpy`, and the final `cudaMemcpy` operations placed outside the `startGpuTimer() / endGpuTimer()` region so only kernels are timed. All GPU numbers use the block size of 512.

**Scan:**

| n | CPU (ms) | Naive (ms) | Work-Efficient (ms) | Thrust (ms) |
|---:|---:|---:|---:|---:|
| 2^10 | 0.0007 | 0.136 | 0.275 | 0.092 |
| 2^14 | 0.008 | 0.178 | 0.331 | 0.112 |
| 2^18 | 0.098 | 0.307 | 0.272 | 0.489 |
| 2^22 | 1.733 | 0.831 | 0.615 | 0.531 |
| 2^25 | 15.41 | 20.95 | 6.591 | 1.360 |

<img alt="image" src="https://github.com/user-attachments/assets/ced4beac-ed56-4735-a0f1-2b0e8528a275" />

**Compaction:**

| n | CPU w/o scan (ms) | CPU w/scan (ms) | Work-Efficient GPU (ms) |
|---:|---:|---:|---:|
| 2^10 | 0.002 | 0.014 | 0.194 |
| 2^14 | 0.029 | 0.062 | 0.182 |
| 2^18 | 0.476 | 1.251 | 0.269 |
| 2^22 | 7.872 | 19.642 | 0.747 |
| 2^25 | 61.37 | 153.81 | 9.685 |

<img alt="image" src="https://github.com/user-attachments/assets/9fdada74-aa30-419d-ac0d-dc76f3dfb8ae" />

At small `n` the CPU is much faster as the GPU implementations pay a fixed kernel-launch overhead per level. For **scan**, the Work-Efficient algorithm surpasses the CPU around `n = 2^18`. For **compaction** this happens earlier, around `n = 2^16`, as GPU compaction does only one scan + a scatter. On the other hand, CPU `compactWithScan` does three serial passes with two host-side allocations. Naive and Work-Efficient have the same order of memory traffic, but Work-Efficient does `O(n)` additions vs Naive's `O(n log n)`, which shows up as a ~3.2x gap at `2^25`. Thrust is also competitive with my Work-Efficient at small-to-mid `n` because I excluded the one-time module/context initialization when testing. At `2^25` it pulls ahead (1.36 ms vs 6.59 ms), which is the expected payoff.


### <ins>What's Happening Inside Thrust?</ins>
Thrust's `exclusive_scan` uses a two-level blocked scan rather than my per-level sweep: each block does a local scan entirely in shared memory and writes out only its block sum. Then a small kernel scans the block sums, and a third kernel adds each block's prefix back to its elements. This is 3–5 kernel launches total regardless of `n`, versus my `2*log_2(n) + 1`. It also moves ~n bytes per pass, not one full pass per level, which is why Thrust pulls ahead of my Work-Efficient at `2^25` (1.36 ms vs 6.59 ms, a 4.8x gap). At lower to medium values of `n` Thrust is competitive with my Work-Efficient implementation because I excluded the one-time module/context initialization (~26 ms in the earlier tests).

<img alt="image" src="https://github.com/user-attachments/assets/dec3bb57-9a11-472c-aa8c-e71bf1d72f0a" />
One `StreamCompaction::Thrust::scan` call at n = 2^22. The scan itself (`thrust::exclusive_scan`, 644 µs) is a small fraction of the surrounding pipeline. The host-to-device `two_system_copy` (1.44 ms) and device-to-host `copy` (1.62 ms) together take ~5x longer than the scan. This is exactly why the assignment excludes memory operations from the timer.


### <ins>Performance Bottlenecks: Memory I/O or Computation?</ins>
The bottleneck shifts with array size:

- **Small n ($\le$ 2^14)**: Both CPU and GPU are fast enough that per-kernel fixed cost dominates, which is why the CPU (zero launches) wins outright up to ~`2^14`. Therefore our bottleneck is the launch overhead. 
- **Mid n (~2^18)**: GPU kernels are doing enough work to make the launch overhead neligible, but are not yet bandwidth-bound.
- **Large n (2^22–2^25)**: Every level of my sweep reads and writes the whole working array, so the total traffic is `O(n log n)` bytes despite `O(n)` adds. Hence we are memory-bound.

Within the large-`n` regime, the implementations differ:

- **Naive scan** pays the same memory traffic as Work-Efficient but with around 25x more additions. This is why it takes 20.95 ms at `2^25` vs Work-Efficient's 6.59 ms and why it's actually slower than the CPU at the same size (15.41 ms). The extra additions per level outweigh the GPU's parallelism advantage.
- **CPU versions** are pure compute-bound. The CPU scan time scales linearly and never catches the GPU once the arrays are big enough.
- **Compaction's scatter step** is bandwidth-bound (non-contiguous writes to `odata[indices[i]]`), but it's a small fraction of total time at large `n` where the scan dominates.
- **Thrust** is the odd one out since its blocked approach reduces both memory traffic and launch count. Therefore it stays launch-limited through mid sizes and only becomes bandwidth-limited at the very top, which is why it beats every other implementation at every size $\ge$ 2^14 in the final table.

<img alt="image" src="https://github.com/user-attachments/assets/3418e833-2636-4995-9216-825f69684360" />

Nsight Systems capture of the compaction tests at n = 2^22. Green bars are host-to-device copies and the red bars are device-to-host copies. Even for memory-bound algorithms, the kernels barely register on the timeline. Most of the time is spent on the transfers.


### <ins>Added Tests and Modifications</ins>
- Added `#include <thrust/copy.h>` to `thrust.cu` as `thrust::copy` wasn't working properly.
- Added a `#define USE_TAIL` toggle in `efficient.cu` to measure the tail optimization side-by-side with the baseline. Submitted version leaves it set to `1`.
- No `CMakeLists.txt` changes.


### <ins>Full Test Output</ins>
Run at `SIZE = 2^22`, `blockSize = 512`, `USE_TAIL = 1`, Release x64.

```
****************
** SCAN TESTS **
****************
    [  49  36  38  26  15  25  29  11  26  37  15  46  16 ...  46   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 1.911ms    (std::chrono Measured)
    [   0  49  85 123 149 164 189 218 229 255 292 307 353 ... 102706625 102706671 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 2.2651ms    (std::chrono Measured)
    [   0  49  85 123 149 164 189 218 229 255 292 307 353 ... 102706562 102706608 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.806272ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.650944ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.742048ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.636512ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.515072ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.43344ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   3   0   0   2   1   3   3   1   2   3   1   0   0 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 6.862ms    (std::chrono Measured)
    [   3   2   1   3   3   1   2   3   1   2   2   2   1 ...   3   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 6.9343ms    (std::chrono Measured)
    [   3   2   1   3   3   1   2   3   1   2   2   2   1 ...   2   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 14.6162ms    (std::chrono Measured)
    [   3   2   1   3   3   1   2   3   1   2   2   2   1 ...   3   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.672192ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.568608ms    (CUDA Measured)
    passed
```
