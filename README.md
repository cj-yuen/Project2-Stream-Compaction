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

Scan has 2 parallel variants on GPU: 
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
Three serial loops. `scan` is the classic single-pass exclusive prefix sum — write the running total, then add the current element. `compactWithoutScan` uses a single write pointer to copy every nonzero element forward in order. `compactWithScan` does map → scan → scatter in three passes, matching the GPU pipeline structurally so I can compare its output against `compactWithoutScan`.

## Part 2: Naive GPU Scan
Each level `d` uses an offset of `2^(d-1)`, and every element `k >= offset` sums itself with the element `offset` positions to its left, for `ilog2ceil(n)` levels. Since threads race when reading and writing the same array, we alternate between two device buffers (`dev_a` ↔ `dev_b`) each level. A final `kernShiftRight` shifts the inclusive result right by one and inserts 0 at index 0 to get the exclusive scan. Works for both power-of-two and non-power-of-two sizes.

## Part 3: Work-Efficient GPU Scan & Stream Compaction
Based on GPU Gems 3, Chapter 39 - [Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html), operating on a power-of-two padded buffer. **Up-sweep** does a parallel reduction up the binary tree, **root-zero** sets the last element to 0, and **down-sweep** traverses back down, passing each node's value to its left child and setting its right child to the sum. Since no thread writes a location another thread reads in the same level, the scan runs fully in place.

Compaction wraps this: `kernMapToBoolean` produces the 0/1 array, `cudaMemcpy` moves it into the padded scan buffer, `scanDevice` exclusive-scans it, and `kernScatter` writes `idata[i]` to `odata[indices[i]]` wherever `bools[i] == 1`. To get the total count, I read back just the last scanned value plus the last input element — one int's worth of D2H traffic instead of a full reduction.

Non-power-of-two arrays are handled by rounding up to `n_pow2` and memset-ing the padding to 0.

## Part 4: Using Thrust's Implementation
A thin wrapper around `thrust::exclusive_scan`. I construct the device input and output vectors *outside* the timer so only the scan call is measured — this excludes the implicit H2D copy when building a `device_vector` from a `host_vector` and the final D2H `thrust::copy`. Added `#include <thrust/copy.h>` to pull in `thrust::copy`.

## Part 5: Tail Optimization (Extra Credit)
The default work-efficient scan launches `2·log₂(n) + 1` kernels, and at the deepest levels `count = n_pow2 >> (d+1)` drops below `blockSize` — so the last few launches are full blocks where nearly every thread is idle. My optimization detects the first level where `count <= blockSize` and collapses all remaining levels of that sweep into a single-block kernel that loops over levels with `__syncthreads()` between them. Down-sweep mirrors this (descending `d`, args swapped). Kernel launches get cut roughly in half.

| n | No tail (ms) | With tail (ms) | Speedup |
|---:|---:|---:|---:|
| 2^10 | 0.650 | 0.275 | **2.36×** |
| 2^14 | 0.400 | 0.331 | **1.21×** |
| 2^18 | 0.494 | 0.272 | **1.82×** |
| 2^22 | 0.959 | 0.615 | **1.56×** |
| 2^25 | 6.342 | 6.591 | 0.96× |

The speedup is largest at small-to-mid `n`, where launch overhead dominates. At `2^25`, the multi-block portion is already doing almost all the work, so the tail saves only ~18 kernel launches out of 51 — under 1% of total time, within run-to-run noise. The multi-run compaction numbers tell the same story: `work-efficient compact` at `2^25` drops from 11.15 ms (no tail) to 9.69 ms (with tail), a ~1.15× win.

---

## Performance Analysis 
### <ins>Block Size Optimization</ins>
Each GPU implementation was tested with the following block sizes `64, 128, 256, 512, 1024` at `n = 2^22` before final data collection, so the tables below compare roughly-optimized implementations rather than unoptimized ones.

| blockSize | Naive (ms) | Efficient (ms) |
|---:|---:|---:|
| 64   | 1.934 | 2.840 |
| 128  | 0.856 | 0.732 |
| 256  | 0.880 | 1.074 |
| 512  | 0.823 | 0.763 |
| 1024 | 1.391 | 0.610 |

<img alt="image" src="https://github.com/user-attachments/assets/c3f8b970-9499-400a-b6dd-f74f79c8962c" />

I chose **blockSize = 512** for both implementations. It's the best Naive time, and works well for Efficient time. Performance across 128-512 was essentially flat, so any of those could work. However, 64 was too small (per-block `__syncthreads()` overhead) and 1024 hurts the memory-bound naive scan.


### <ins>GPU vs CPU Scan Comparison</ins>
All measurements in **Release x64** on an RTX 4090 Laptop GPU, V-Sync off. Data collected by sweeping the test harness across five sizes (`SIZE = 2^10, 2^14, 2^18, 2^22, 2^25`) in a single process. Every GPU implementation has `cudaMalloc`, `cudaMemset`, H2D `cudaMemcpy`, and the final D2H `cudaMemcpy` placed **outside** the `startGpuTimer() / endGpuTimer()` region — only kernels are timed. All GPU numbers use the block size chosen in Q1 (512).

**Scan:**

| n | CPU (ms) | Naive (ms) | Efficient (ms) | Thrust (ms) |
|---:|---:|---:|---:|---:|
| 2^10 | 0.0007 | 0.136 | 0.275 | 0.092 |
| 2^14 | 0.008 | 0.178 | 0.331 | 0.112 |
| 2^18 | 0.098 | 0.307 | 0.272 | 0.489 |
| 2^22 | 1.733 | 0.831 | 0.615 | 0.531 |
| 2^25 | 15.41 | 20.95 | 6.591 | 1.360 |

<img alt="image" src="https://github.com/user-attachments/assets/ced4beac-ed56-4735-a0f1-2b0e8528a275" />

**Compaction:**

| n | CPU w/o scan (ms) | CPU w/ scan (ms) | Efficient GPU (ms) |
|---:|---:|---:|---:|
| 2^10 | 0.002 | 0.014 | 0.194 |
| 2^14 | 0.029 | 0.062 | 0.182 |
| 2^18 | 0.476 | 1.251 | 0.269 |
| 2^22 | 7.872 | 19.642 | 0.747 |
| 2^25 | 61.37 | 153.81 | 9.685 |

<img alt="image" src="https://github.com/user-attachments/assets/9fdada74-aa30-419d-ac0d-dc76f3dfb8ae" />

At small `n` the CPU wins outright. The GPU implementations pay a fixed kernel-launch overhead per level that swamps the actual work. For **scan**, the CPU and Efficient crossover is around `n = 2^18`; for **compaction** it happens earlier, around `n = 2^16`, because GPU compaction does only one scan plus a scatter while CPU `compactWithScan` does three serial passes with two host-side allocations. Naive and Efficient have the same order of memory traffic, but Efficient does `O(n)` additions vs Naive's `O(n log n)`, which shows up as a ~3.2× gap at `2^25`. Thrust is now competitive with my Efficient at small-to-mid `n` because running all five sizes in a single process amortizes its one-time module/context initialization. At `2^25` it pulls ahead (1.36 ms vs 6.59 ms), which is the expected payoff of its blocked two-level scan: only 3–5 kernel launches total versus my `2*log_2(n) + 1` (51 launches at that size), and it moves `O(n)` bytes per pass instead of one full pass per level.


### <ins>What's Happening Inside Thrust?</ins>
Thrust's `exclusive_scan` uses a **two-level blocked scan** rather than my per-level sweep: each block does a local scan entirely in shared memory and writes out only its block sum, then a small kernel scans the block sums, then a third kernel adds each block's prefix back to its elements. This is 3–5 kernel launches total regardless of `n`, versus my `2*log_2(n) + 1` (51 launches at `2^25`). It also moves ~O(n) bytes per pass, not one full pass per level, which is why Thrust pulls ahead of my Efficient at `2^25` (1.36 ms vs 6.59 ms — a 4.8× gap). At small-to-mid `n` Thrust is competitive with my Efficient because running all five sizes in a single process amortizes its one-time module/context initialization (~26 ms in the earlier per-process runs). *(I didn't need Nsight to explain this — the timing shape across sizes is enough to see that Thrust's cost grows far more slowly with `n` than mine.)*


### <ins>Performance Bottlenecks: Memory I/O or Computation?</ins>
- **Small n (≤ 2^14)**: launch overhead. Both CPU and GPU are fast enough that per-kernel fixed cost dominates, which is why the CPU (zero launches) wins outright up to ~`2^14`.
- **Mid n (2^18)**: crossover region for Efficient vs CPU. GPU kernels are now doing enough work to amortize launches, but not yet bandwidth-bound.
- **Large n (2^22–2^25)**: memory bandwidth. Every level of my sweep reads and writes the whole working array, so the total traffic is `O(n log n)` bytes despite `O(n)` adds. Naive scan pays the same traffic *plus* ~25× more additions, showing up as Naive taking 20.95 ms at `2^25` vs Efficient's 6.59 ms.
- **Naive at 2^25** (20.95 ms) is actually *slower* than the CPU at 2^25 (15.41 ms) — the extra additions per level outweigh the GPU's parallelism advantage at that size.
- **Compaction's scatter step** is bandwidth-bound (non-contiguous writes to `odata[indices[i]]`), but it's a small fraction of total time at large `n` where the scan dominates.
- **CPU versions** are pure compute-bound — the serial running-sum dependency prevents ILP, so CPU scan time scales linearly and never catches the GPU once the arrays are big enough.
- **Thrust** is the odd one out: its blocked approach reduces memory traffic and launch count, so it stays launch-overhead-limited through mid sizes and bandwidth-limited only at the very top. That's why it beats every other implementation at every size ≥ 2^14 in the final table.


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
