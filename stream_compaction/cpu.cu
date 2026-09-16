#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO

            // exclusive prefix sum
            int sum = 0;
            for (int i = 0; i < n; ++i) {
                odata[i] = sum;
                sum += idata[i];
            }

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO

            int count = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[count++] = idata[i];
                }
            }

            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO

            // step 1: temp boolean array
            int* temp = new int[n];
            for (int i = 0; i < n; ++i) {
                temp[i] = (idata[i] != 0) ? 1 : 0;
            }

            // step 2: exclusive scan
            int* scan = new int[n];
            int sum = 0;
            for (int i = 0; i < n; ++i) {
                scan[i] = sum;
                sum += temp[i];
            }

            // step 3: scatter 
            for (int i = 0; i < n; ++i) {
                if (temp[i]) {
                    odata[scan[i]] = idata[i];
                }
            }

            int count = sum;
            delete[] temp;
            delete[] scan;

            timer().endCpuTimer();
            return count;
        }
    }
}
