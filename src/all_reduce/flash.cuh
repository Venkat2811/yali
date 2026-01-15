/*************************************************************************
 * Low-latency AllReduce kernel for small messages.
 *
 * This kernel uses cp.async prefetching and staged double-buffering to
 * minimize latency for small AllReduce operations over NVLink.
 *
 * Extracted from main.cu as part of Yali code reorganization.
 ************************************************************************/

#ifndef YALI_ALL_REDUCE_LOWLAT_CUH_
#define YALI_ALL_REDUCE_LOWLAT_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdint.h>

#include "../kernels/type_ops.cuh"
#include "yali_launch.h"

namespace yali {

namespace {

template <typename T>
__device__ inline void reduce_scalar_chunk(const T* localPtr, const T* peerPtr, T* outPtr, size_t count) {
    for (size_t idx = threadIdx.x; idx < count; idx += blockDim.x) {
        outPtr[idx] = yali::ValueOps<T>::Add(localPtr[idx], peerPtr[idx]);
    }
}

}  // namespace

// Low-latency AllReduce kernel with prefetch staging.
// Template parameters:
//   T - Data type (float, __half, __nv_bfloat16)
//   PrefetchStages - Number of prefetch stages (typically 3)
//
// The kernel uses cp.async for asynchronous memory copies when available
// (sm_80-sm_90), with fallback to regular loads for older/newer architectures.
template <typename T, int PrefetchStages>
__global__ void FlashKernel(const YaliLaunchArgs* argsArray, int laneCount, int ctasPerLane) {
    static_assert(16 % sizeof(T) == 0, "Vector chunk must be 16-byte aligned");
    constexpr int kVectorBytes = 16;
    constexpr int kVectorWidth = kVectorBytes / sizeof(T);
    using StageVec = uint4;

    const int blockId = blockIdx.x;
    const int lane = blockId % laneCount;
    const int laneCta = blockId / laneCount;
    if (lane >= laneCount || laneCta >= ctasPerLane)
        return;

    YaliLaunchArgs launchArgs = argsArray[lane];
    if (launchArgs.elementCount == 0)
        return;

    const size_t totalElems = launchArgs.elementCount;
    const size_t elemsPerCta = (totalElems + static_cast<size_t>(ctasPerLane) - 1) / static_cast<size_t>(ctasPerLane);
    const size_t startElem = min(static_cast<size_t>(laneCta) * elemsPerCta, totalElems);
    const size_t endElem = min(startElem + elemsPerCta, totalElems);
    if (startElem >= endElem)
        return;

    const size_t sendBaseOffset = launchArgs.sendOffset / sizeof(T);
    const size_t recvBaseOffset = launchArgs.recvOffset / sizeof(T);
    const T* localBase = reinterpret_cast<const T*>(launchArgs.localInput) + sendBaseOffset + startElem;
    const T* peerBase = reinterpret_cast<const T*>(launchArgs.peerInput) + sendBaseOffset + startElem;
    T* outBase = reinterpret_cast<T*>(launchArgs.localOutput) + recvBaseOffset + startElem;
    const size_t chunkElems = endElem - startElem;

    const uintptr_t addrMask = reinterpret_cast<uintptr_t>(localBase) | reinterpret_cast<uintptr_t>(peerBase) |
                               reinterpret_cast<uintptr_t>(outBase);
    const bool vectorizable = ((addrMask) & 0xF) == 0;
    if (!vectorizable || chunkElems < 32) {
        reduce_scalar_chunk(localBase, peerBase, outBase, chunkElems);
        return;
    }

    constexpr int kBuffers = PrefetchStages;
    const int stageVecs = blockDim.x;
    const int stageElems = stageVecs * kVectorWidth;

    extern __shared__ char sharedRaw[];
    StageVec* peerStages = reinterpret_cast<StageVec*>(sharedRaw);

    int vecCounts[kBuffers] = {0};
    size_t tailCounts[kBuffers] = {0};
    size_t stageOffsets[kBuffers] = {0};
    bool stageHasPayload[kBuffers] = {false};

    const int numStages = static_cast<int>((chunkElems + stageElems - 1) / stageElems);
    if (numStages == 0)
        return;

    auto prefetchStage = [&](int stageIdx, int bufIdx) {
        const size_t stageOffset = static_cast<size_t>(stageIdx) * stageElems;
        stageOffsets[bufIdx] = stageOffset;
        const size_t remaining = min(static_cast<size_t>(stageElems), chunkElems - stageOffset);
        const int vecCount = static_cast<int>(remaining / kVectorWidth);
        const size_t tail = remaining - static_cast<size_t>(vecCount) * kVectorWidth;
        vecCounts[bufIdx] = vecCount;
        tailCounts[bufIdx] = tail;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDA_ARCH__ < 1000)
        // cp.async is supported on sm_80-sm_90 (Ampere, Hopper)
        // For sm_100+ (Blackwell), use regular loads as cp.async semantics changed
        if (vecCount > 0) {
            const char* peerSrcBase = reinterpret_cast<const char*>(peerBase + stageOffset);
            StageVec* peerBuf = peerStages + bufIdx * stageVecs;
            for (int vec = threadIdx.x; vec < vecCount; vec += blockDim.x) {
                char* dst = reinterpret_cast<char*>(peerBuf + vec);
                const char* src = peerSrcBase + static_cast<size_t>(vec) * kVectorBytes;
                // Use __cvta_generic_to_shared to get 32-bit shared address for inline asm
                // This is required for CUDA 13+ which uses 64-bit generic pointers
                unsigned int dst32 = static_cast<unsigned int>(__cvta_generic_to_shared(dst));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(dst32), "l"(src));
            }

            asm volatile("cp.async.commit_group;\n" ::);
        }
#else
        if (vecCount > 0) {
            StageVec* peerBuf = peerStages + bufIdx * stageVecs;
            const StageVec* peerSrc = reinterpret_cast<const StageVec*>(peerBase + stageOffset);
            for (int vec = threadIdx.x; vec < vecCount; vec += blockDim.x) {
                peerBuf[vec] = peerSrc[vec];
            }
        }
#endif
        stageHasPayload[bufIdx] = vecCount > 0;
    };

    const int initialPrefetch = min(PrefetchStages, numStages);
    for (int preload = 0; preload < initialPrefetch; ++preload) {
        prefetchStage(preload, preload % PrefetchStages);
    }

    for (int stage = 0; stage < numStages; ++stage) {
        const int curBuf = stage % PrefetchStages;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        if (stageHasPayload[curBuf]) {
            if (stage + PrefetchStages < numStages) {
                asm volatile("cp.async.wait_group %0;\n" : : "n"(PrefetchStages - 1));
            } else {
                asm volatile("cp.async.wait_group 0;\n" ::);
            }
        }
#endif
        __syncthreads();

        const size_t stageOffset = stageOffsets[curBuf];
        const int vecCount = vecCounts[curBuf];
        const size_t tail = tailCounts[curBuf];
        StageVec* peerVec = peerStages + curBuf * stageVecs;
        T* outVec = outBase + stageOffset;
        const T* localVecBase = localBase + stageOffset;

        for (int vec = threadIdx.x; vec < vecCount; vec += blockDim.x) {
            T* outVals = outVec + static_cast<size_t>(vec) * kVectorWidth;
            const T* localVals = localVecBase + static_cast<size_t>(vec) * kVectorWidth;
            const T* peerVals = reinterpret_cast<const T*>(peerVec + vec);
#pragma unroll
            for (int i = 0; i < kVectorWidth; ++i) {
                outVals[i] = yali::ValueOps<T>::Add(localVals[i], peerVals[i]);
            }
        }
        __syncthreads();

        const size_t vectorizedElems = static_cast<size_t>(vecCount) * kVectorWidth;
        for (size_t idx = threadIdx.x; idx < tail; idx += blockDim.x) {
            const size_t elemIdx = stageOffset + vectorizedElems + idx;
            outBase[elemIdx] = localBase[elemIdx] + peerBase[elemIdx];
        }
        __syncthreads();

        // Prefetch the next stage after we're done with the current one
        const int nextStage = stage + PrefetchStages;
        if (nextStage < numStages) {
            prefetchStage(nextStage, nextStage % PrefetchStages);
        }
    }
}

// Configuration constants for the low-latency kernel
struct FlashConfig {
    static constexpr int kDefaultBlockSize = 512;
    static constexpr int kDefaultPrefetchStages = 3;

    // Calculate shared memory requirement
    static size_t SharedMemoryBytes(int blockSize, int prefetchStages, size_t elemSize) {
        // 16 bytes per vector, blockSize vectors per stage, prefetchStages stages
        return static_cast<size_t>(blockSize) * static_cast<size_t>(prefetchStages) * 16u;
    }

    // Calculate elements processed per tile
    static size_t TileElements(int blockSize, int prefetchStages, size_t elemSize) {
        const int vectorElems = 16 / static_cast<int>(elemSize);
        return static_cast<size_t>(blockSize) * static_cast<size_t>(prefetchStages) * static_cast<size_t>(vectorElems);
    }
};

}  // namespace yali

#endif  // YALI_ALL_REDUCE_LOWLAT_CUH_
