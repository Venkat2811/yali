/*************************************************************************
 * Yali persistent all-reduce kernel (bandwidth-optimized).
 * Supports SUM on float/fp16/bf16 buffers.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "device.h"
#include "primitives.h"
#include "stream.cuh"
#include "yali_launch.h"

namespace {
__device__ inline yali::RingConfig make_ring_config(uint64_t* seq, uint64_t* gating, char* data, int32_t capacity,
                                                    int32_t slotBytes, int32_t slotStride) {
    yali::RingConfig cfg;
    cfg.capacity = capacity;
    cfg.slotBytes = slotBytes;
    cfg.slotStride = slotStride;
    cfg.sequence = seq;
    cfg.gating = gating;
    cfg.data = data;
    return cfg;
}

template <typename T>
__device__ inline void run_with_op(const YaliLaunchArgs& launchArgs, yali::PersistentArgs args) {
    FuncSum<T> op;
    yali::allreduce_persistent<T, FuncSum<T>>(args, op);
}

}  // namespace

__device__ inline void yali_kernel_entry(YaliLaunchArgs launchArgs) {
    if (launchArgs.debugEarlyExit) {
        if (blockIdx.x == 0 && threadIdx.x == 0 && launchArgs.elementCount > 0) {
            if (launchArgs.datatype == ncclHalf) {
                __half* outHalf =
                    reinterpret_cast<__half*>(static_cast<char*>(launchArgs.localOutput) + launchArgs.recvOffset);
                const __half* inHalf = reinterpret_cast<const __half*>(static_cast<const char*>(launchArgs.localInput) +
                                                                       launchArgs.sendOffset);
                if (outHalf && inHalf) {
                    outHalf[0] = inHalf[0];
                }
            } else if (launchArgs.datatype == ncclBfloat16) {
                __nv_bfloat16* outBf = reinterpret_cast<__nv_bfloat16*>(static_cast<char*>(launchArgs.localOutput) +
                                                                        launchArgs.recvOffset);
                const __nv_bfloat16* inBf = reinterpret_cast<const __nv_bfloat16*>(
                    static_cast<const char*>(launchArgs.localInput) + launchArgs.sendOffset);
                if (outBf && inBf) {
                    outBf[0] = inBf[0];
                }
            } else {
                float* outFloat =
                    reinterpret_cast<float*>(static_cast<char*>(launchArgs.localOutput) + launchArgs.recvOffset);
                const float* inFloat = reinterpret_cast<const float*>(static_cast<const char*>(launchArgs.localInput) +
                                                                      launchArgs.sendOffset);
                if (outFloat && inFloat) {
                    outFloat[0] = inFloat[0];
                }
            }
        }
        return;
    }

    int32_t sendStride = launchArgs.sendSlotStride > 0 ? launchArgs.sendSlotStride : launchArgs.sendSlotBytes;
    int32_t recvStride = launchArgs.recvSlotStride > 0 ? launchArgs.recvSlotStride : launchArgs.recvSlotBytes;

    yali::PersistentArgs args;
    args.rank = launchArgs.rank;
    args.sendCfg = make_ring_config(launchArgs.sendSequence, launchArgs.sendGating, launchArgs.sendData,
                                    launchArgs.sendCapacity, launchArgs.sendSlotBytes, sendStride);
    args.recvCfg = make_ring_config(launchArgs.recvSequence, launchArgs.recvGating, launchArgs.recvData,
                                    launchArgs.recvCapacity, launchArgs.recvSlotBytes, recvStride);
    args.localInput = launchArgs.localInput;
    args.localOutput = launchArgs.localOutput;
    args.elementCount = launchArgs.elementCount;
    args.elementSize = launchArgs.elementSize;
    args.initialSeq = launchArgs.initialSequence;
    args.sendOffset = launchArgs.sendOffset;
    args.recvOffset = launchArgs.recvOffset;

    if (launchArgs.redOp != ncclSum) {
        return;
    }

    if (launchArgs.datatype == ncclFloat && launchArgs.elementSize == sizeof(float)) {
        run_with_op<float>(launchArgs, args);
        return;
    }

    if (launchArgs.datatype == ncclHalf && launchArgs.elementSize == sizeof(__half)) {
        run_with_op<__half>(launchArgs, args);
        return;
    }

    if (launchArgs.datatype == ncclBfloat16 && launchArgs.elementSize == sizeof(__nv_bfloat16)) {
        run_with_op<__nv_bfloat16>(launchArgs, args);
        return;
    }
}

extern "C" __global__ void YaliKernel(YaliLaunchArgs launchArgs) {
    yali_kernel_entry(launchArgs);
}

// Provide an underscore-prefixed symbol for compatibility.
extern "C" __global__ void _YaliKernel(YaliLaunchArgs launchArgs) {
    yali_kernel_entry(launchArgs);
}
