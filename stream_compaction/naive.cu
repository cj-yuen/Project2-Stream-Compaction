#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernNaiveScan(int n, int d, int *odata, const int *idata) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n) {
                return;
            }

            int offset = 1 << (d - 1);
            odata[index] = (index >= offset) ? idata[index - offset] + idata[index] : idata[index];
        }

        __global__ void kernShiftRight(int n, int* odata, const int* idata) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n) {
                return;
            }

            odata[index] = (index == 0) ? 0 : idata[index - 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            
            if (n <= 0) {
                return;
            }

            int* dev_a, * dev_b, * dev_out;
			cudaMalloc((void**)&dev_a, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_a failed!");

			cudaMalloc((void**)&dev_b, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_b failed!");

			cudaMalloc((void**)&dev_out, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_out failed!");

			cudaMemcpy(dev_a, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_a failed!");

            timer().startGpuTimer();

            int numLevels = ilog2ceil(n);
            const int blockSize = 512;
			const int gridSize = (n + blockSize - 1) / blockSize;

            int* readFrom = dev_a;
            int* writeTo = dev_b;

            for (int d = 1; d <= numLevels; d++) {
                kernNaiveScan<<<gridSize, blockSize>>>(n, d, writeTo, readFrom);
                checkCUDAError("kernNaiveScan failed!");
                
                // Swap readFrom and writeTo
                int* temp = readFrom;
                readFrom = writeTo;
                writeTo = temp;
			}

            // inclusive --> exclusive 
			kernShiftRight<<<gridSize, blockSize>>>(n, dev_out, readFrom);
			checkCUDAError("kernShiftRight failed!");

            timer().endGpuTimer();

			cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_out failed!");
			
            cudaFree(dev_a);
			checkCUDAError("cudaFree dev_a failed!");

			cudaFree(dev_b);
			checkCUDAError("cudaFree dev_b failed!");

			cudaFree(dev_out);
			checkCUDAError("cudaFree dev_out failed!");
        }
    }
}
