/*************************************************************************
 * Bandwidth-optimized AllReduce kernel for large messages.
 *
 * This kernel uses a persistent ring-based approach with slot-based
 * flow control to maximize throughput for large AllReduce operations.
 *
 * Implementation is in src/kernels/stream.cuh and stream.cu.
 ************************************************************************/

#ifndef YALI_ALL_REDUCE_BANDWIDTH_CUH_
#define YALI_ALL_REDUCE_BANDWIDTH_CUH_

// The stream kernel implementation is in src/kernels/stream.cuh
// which contains:
//   - yali::PersistentArgs - argument bundle
//   - yali::allreduce_persistent<T, RedOp>() - the main kernel logic
//
// The entry point is in src/kernels/stream.cu:
//   - extern "C" __global__ void YaliKernel(YaliLaunchArgs)
//   - extern "C" __global__ void _YaliKernel(YaliLaunchArgs)

#include "../kernels/stream.cuh"
#include "yali_launch.h"

namespace yali {

// Forward declaration of the stream kernel entry point
// Defined in src/kernels/stream.cu
extern "C" __global__ void YaliKernel(YaliLaunchArgs launchArgs);
extern "C" __global__ void _YaliKernel(YaliLaunchArgs launchArgs);

// Configuration constants for the stream kernel
struct StreamConfig {
    static constexpr int kDefaultBlockSize = 1024;

    // The stream kernel uses a ring-based approach with these defaults:
    // - Multiple slots for pipelining
    // - Slot size determined by tuning heuristics
    // - Flow control via sequence numbers
};

}  // namespace yali

#endif  // YALI_ALL_REDUCE_BANDWIDTH_CUH_
