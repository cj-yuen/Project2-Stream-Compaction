#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define USE_TAIL 1  // 0 --> disables tail optimization

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		// kernels for upsweep & downsweep
        __global__ void kernUpsweep(int count, int d, int* data) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx >= count) {
                return;
            }

            int pos = (idx << (d + 1)) + (1 << (d + 1)) - 1;
            int stride = 1 << d;
            data[pos] += data[pos - stride];
        }

        __global__ void kernDownsweep(int count, int d, int* data) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx >= count) {
                return;
            }

            int pos = (idx << (d + 1)) + (1 << (d + 1)) - 1;
            int stride = 1 << d;

            int left = data[pos - stride];
            data[pos - stride] = data[pos];
            data[pos] += left;
        }

        __global__ void kernZero(int idx, int* data) {
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                data[idx] = 0;
            }
        }

        __global__ void kernUpsweepTail(int n_pow2, int startD, int numLevels, int* data) {
            for (int d = startD; d < numLevels; ++d) {
                int count = n_pow2 >> (d + 1);
                if (threadIdx.x < count) {
                    int pos = (threadIdx.x << (d + 1)) + (1 << (d + 1)) - 1;
                    int stride = 1 << d;
                    data[pos] += data[pos - stride];
                }

                __syncthreads();
            }
        }

        __global__ void kernDownsweepTail(int n_pow2, int startD, int endD, int* data) {
            for (int d = startD; d >= endD; --d) {
                int count = n_pow2 >> (d + 1);
                if (threadIdx.x < count) {
                    int pos = (threadIdx.x << (d + 1)) + (1 << (d + 1)) - 1;
                    int stride = 1 << d;
                    int left = data[pos - stride];
                    data[pos - stride] = data[pos];
                    data[pos] += left;
                }

                __syncthreads();
            }
		}

        // device exclusive scan of dev_data in place
        void scanDevice(int n_pow2, int* dev_data) {
            int numLevels = ilog2ceil(n_pow2);
            const int blockSize = 512;

            if (numLevels == 0) {
				kernZero<<<1, 1 >>>(0, dev_data);
                checkCUDAError("kernZero failed!");
				return;
            }

            // find 1st level w/ count < blockSize
            int cutoffD = numLevels;
            for (int d = 0; d < numLevels; ++d) {
                int count = n_pow2 >> (d + 1);
                if (count <= blockSize) {
                    cutoffD = d;
                    break;
                }
			}

            // upsweep 
            int upStart = USE_TAIL ? cutoffD : numLevels;
            for (int d = 0; d < upStart; ++d) {
                int count = n_pow2 >> (d + 1);
                int gridSize = (count + blockSize - 1) / blockSize;
                
				kernUpsweep<<<gridSize, blockSize>>>(count, d, dev_data);
				checkCUDAError("kernUpsweep failed!");
            }

#if USE_TAIL
            if (cutoffD < numLevels) {
				kernUpsweepTail<<<1, blockSize>>>(n_pow2, cutoffD, numLevels, dev_data);
				checkCUDAError("kernUpsweepTail failed!");
            }
#endif

            // set root --> 0
            kernZero<<<1, 1 >>>(n_pow2 - 1, dev_data);
			checkCUDAError("kernZero failed!");

            // downsweep
#if USE_TAIL
            if (cutoffD < numLevels) {
				kernDownsweepTail<<<1, blockSize>>>(n_pow2, numLevels - 1, cutoffD, dev_data);
				checkCUDAError("kernDownsweepTail failed!");
            }
#endif 
			int downStop = USE_TAIL ? cutoffD - 1 : numLevels - 1;
            for (int d = downStop; d >= 0; --d) {
				int count = n_pow2 >> (d + 1);
                int gridSize = (count + blockSize - 1) / blockSize;

				kernDownsweep<<<gridSize, blockSize>>>(count, d, dev_data);
				checkCUDAError("kernDownsweep failed!");
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO

            if (n <= 0) {
                return;
            }

            int n_pow2 = 1;
            while (n_pow2 < n) {
                n_pow2 <<= 1;
            }

            int* dev_data;
			cudaMalloc((void**)&dev_data, n_pow2 * sizeof(int));
			checkCUDAError("cudaMalloc dev_data failed!");

			cudaMemset(dev_data, 0, n_pow2 * sizeof(int));
			checkCUDAError("cudaMemset dev_data failed!");

			cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy idata to dev_data failed!");

            timer().startGpuTimer();
			scanDevice(n_pow2, dev_data);
            timer().endGpuTimer();

			cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_data to odata failed!");

			cudaFree(dev_data);
			checkCUDAError("cudaFree dev_data failed!");
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            // TODO

            if (n <= 0) {
                return 0;
            }

            int n_pow2 = 1;
            while (n_pow2 < n) {
                n_pow2 <<= 1;
            }

			int* dev_idata, * dev_bools, * dev_indices, * dev_odata;
			cudaMalloc((void**)&dev_idata, n_pow2 * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");

			cudaMalloc((void**)&dev_bools, n_pow2 * sizeof(int));
			checkCUDAError("cudaMalloc dev_bools failed!");

			cudaMalloc((void**)&dev_indices, n_pow2 * sizeof(int));
			checkCUDAError("cudaMalloc dev_indices failed!");

			cudaMalloc((void**)&dev_odata, n_pow2 * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");

			cudaMemset(dev_indices, 0, n_pow2 * sizeof(int));
			checkCUDAError("cudaMemset dev_indices failed!");

			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy idata to dev_idata failed!");

            const int blockSize = 512;
            const int gridSize = (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();

            // step 1: map idata to bools
			StreamCompaction::Common::kernMapToBoolean<<<gridSize, blockSize>>>(n, dev_bools, dev_idata);
			checkCUDAError("kernMapToBoolean failed!");

            // move bools --> scan buffer
			cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcpy dev_bools to dev_indices failed!");

            // step 2: exclusive scan (in place on buffer)
			scanDevice(n_pow2, dev_indices);

            // step 3: scatter
			StreamCompaction::Common::kernScatter<<<gridSize, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
			checkCUDAError("kernScatter failed!");

            timer().endGpuTimer();

            int lastScanned = 0;
			cudaMemcpy(&lastScanned, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy lastScanned failed!");

			int total = lastScanned + ((idata[n - 1] != 0) ? 1 : 0);

            if (total > 0) {
				cudaMemcpy(odata, dev_odata, total * sizeof(int), cudaMemcpyDeviceToHost);
				checkCUDAError("cudaMemcpy dev_odata to odata failed!");
            }

			cudaFree(dev_idata);
			checkCUDAError("cudaFree dev_idata failed!");

			cudaFree(dev_bools);
			checkCUDAError("cudaFree dev_bools failed!");

			cudaFree(dev_indices);
			checkCUDAError("cudaFree dev_indices failed!");

			cudaFree(dev_odata);
			checkCUDAError("cudaFree dev_odata failed!");

            return total;
        }
    }
}
