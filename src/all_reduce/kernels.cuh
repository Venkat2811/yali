/*************************************************************************
 * Yali AllReduce Kernels - Unified Interface
 *
 * This header provides a unified interface to the Yali AllReduce kernels:
 *   - FlashKernel: For small messages, uses cp.async prefetching
 *   - StreamKernel: For large messages, uses persistent ring approach
 *
 * The kernel selection is typically done by the tuning heuristics in
 * include/yali_tuning.h based on message size and data type.
 ************************************************************************/

#ifndef YALI_ALL_REDUCE_KERNELS_CUH_
#define YALI_ALL_REDUCE_KERNELS_CUH_

// Low-latency kernel (header-only, template-based)
#include "flash.cuh"

// Bandwidth kernel (wraps device/yali_all_reduce.cu)
#include "stream.cuh"

// Public interface types
#include "all_reduce.h"

namespace yali {

// Kernel launcher helper that selects and launches the appropriate kernel.
// This is a convenience wrapper - the harness can also launch kernels directly.
//
// Returns cudaError_t from the kernel launch.
//
// Parameters:
//   - argsDevice: Device pointer to YaliLaunchArgs array (one per lane)
//   - laneCount: Number of lanes
//   - ctasPerLane: CTAs per lane for low-latency kernel
//   - useFlash: true for low-latency kernel, false for stream kernel
//   - dtype: Data type (Float32, Float16, BFloat16)
//   - stream: CUDA stream for kernel launch
//
// NOTE: For stream kernel, launch _YaliKernel directly with single-lane args.
// This helper is mainly for low-latency kernel convenience.
template <typename T, int PrefetchStages = 3>
inline cudaError_t LaunchFlashKernel(const YaliLaunchArgs* argsDevice, int laneCount, int ctasPerLane,
                                     size_t sharedMemBytes, cudaStream_t stream) {
    const unsigned int totalBlocks = static_cast<unsigned int>(laneCount * ctasPerLane);
    if (totalBlocks == 0u) {
        return cudaSuccess;  // Nothing to do
    }

    const dim3 grid(totalBlocks, 1, 1);
    const dim3 block(FlashConfig::kDefaultBlockSize, 1, 1);

    yali::FlashKernel<T, PrefetchStages><<<grid, block, sharedMemBytes, stream>>>(argsDevice, laneCount, ctasPerLane);

    return cudaGetLastError();
}

}  // namespace yali

#endif  // YALI_ALL_REDUCE_KERNELS_CUH_
