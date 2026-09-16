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
| 2^10 | 0.938 | 0.349 | **2.69x** |
| 2^14 | 0.497 | 0.360 | **1.38x** |
| 2^18 | 0.524 | 0.393 | **1.33x** |
| 2^22 | 1.129 | 0.797 | **1.42x** |
| 2^25 | 9.793 | 6.250 | **1.57x** |

---

## Performance Analysis 
### <ins>Block Size Optimization</ins>


### <ins>GPU vs CPU Scan Comparison</ins>
All measurements in **Release x64** on an RTX 4090 Laptop GPU, V-Sync off. Data collected by sweeping the test harness across five sizes (`SIZE = 2^10, 2^14, 2^18, 2^22, 2^25`) in a single process.

| n | CPU (ms) | Naive (ms) | Efficient (ms) | Thrust (ms) |
|---:|---:|---:|---:|---:|
| 2^10 | 0.0018 | 0.142 | 0.349 | 26.05 |
| 2^14 | 0.0137 | 0.157 | 0.360 | 27.37 |
| 2^18 | 0.217 | 0.364 | 0.393 | 26.86 |
| 2^22 | 3.32 | 2.06 | 0.797 | 45.48 |
| 2^25 | 36.87 | 26.21 | 6.25 | 125.18 |

<img alt="image" src="https://github.com/user-attachments/assets/424d5dc7-eb7d-4870-a817-ab0871dc2ceb" />

**Brief explanation of the phenomena:** At small `n` the CPU wins outright — the GPU implementations pay a fixed kernel-launch overhead per level that swamps the actual work. The crossover between CPU and Efficient is around `n = 2^20`. Naive and Efficient have the same order of memory traffic, but Efficient does `O(n)` additions vs Naive's `O(n log n)`, which shows up as a 4.2× gap at `2^25`.


### <ins>What's Happening Inside Thrust?</ins>


### <ins>Performance Bottlenecks: Memory I/O or Computation?</ins>

### <ins>Full Test Output</ins>
