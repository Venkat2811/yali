/*************************************************************************
 * Copyright (c) 2025
 * All rights reserved.
 *
 * Bandwidth-optimized all-reduce kernel for large transfers.
 * Uses ring buffer protocol with vectorized copy/reduce operations.
 *
 * CONSTRAINTS AND LIMITATIONS:
 *
 * 1. ABA Problem / Sequence Number Safety:
 *    The ring buffer uses 64-bit sequence numbers for producer/consumer
 *    coordination. While 64-bit overflow is practically impossible during
 *    a single AllReduce operation (would require 2^64 slots), the current
 *    design assumes:
 *    - Each kernel launch starts with initialSeq from caller
 *    - Sequence numbers increment monotonically within a single launch
 *    - Multiple AllReduce operations should reset sequence base between calls
 *
 *    If sequence numbers were to wrap (ABA problem), the consumer could
 *    misinterpret a new slot as an old completed one. The host harness
 *    (main_mpi.cu) correctly tracks sequenceBase and increments it between
 *    iterations.
 *
 * 2. Two-GPU Only Architecture:
 *    This kernel is designed for a bipartite 2-GPU topology:
 *    - Each rank has ONE sendCfg (to peer) and ONE recvCfg (from peer)
 *    - Extending to N>2 GPUs requires a ring topology redesign with
 *      sendCfg[N-1] and recvCfg[N-1] arrays
 *
 * 3. Cross-Process IPC Visibility:
 *    The kernel relies on __threadfence_system() for memory ordering.
 *    This works for:
 *    - Single-process multi-GPU (always correct)
 *    - Cross-process via CUDA IPC on NVLink (strong ordering masks issues)
 *    For maximum safety, the host-side code uses cudaDeviceSynchronize()
 *    before MPI barriers to ensure cross-process visibility.
 ************************************************************************/

#ifndef YALI_KERNELS_BANDWIDTH_CUH_
#define YALI_KERNELS_BANDWIDTH_CUH_

#include <stddef.h>
#include <stdint.h>

#include "collectives.h"
#include "device.h"
#include "primitives.h"
#include "ring.cuh"
#include "type_ops.cuh"

namespace yali {

// Argument bundle for persistent kernel.
struct PersistentArgs {
    int rank;
    RingConfig sendCfg;
    RingConfig recvCfg;
    void* localInput;
    void* localOutput;
    size_t elementCount;
    int elementSize;
    uint64_t initialSeq;
    uint64_t sendOffset;
    uint64_t recvOffset;
};

// Persistent kernel skeleton that stages payloads through a Yali-style ring.
// The producer pushes local chunks into the outbound ring, while the consumer
// waits for inbound payloads and reduces them into the destination buffer.
template <typename T, typename RedOp>
__device__ void allreduce_persistent(PersistentArgs args, RedOp redop) {
    auto min_bytes = []
        __device__(size_t a, int32_t b) -> int32_t { return static_cast<int32_t>(a < static_cast<size_t>(b) ? a : b); };

    const size_t totalBytes = args.elementCount * static_cast<size_t>(args.elementSize);
    T* outBase = reinterpret_cast<T*>(reinterpret_cast<char*>(args.localOutput) + args.recvOffset);
    const T* inBase = reinterpret_cast<const T*>(reinterpret_cast<const char*>(args.localInput) + args.sendOffset);
    size_t totalElems = args.elementCount;

    for (size_t idx = threadIdx.x; idx < totalElems; idx += blockDim.x) {
        outBase[idx] = inBase[idx];
    }
    __syncthreads();

    uint64_t nextProduce = args.initialSeq;
    uint64_t nextConsume = args.initialSeq;
    size_t producedBytes = 0;
    size_t consumedBytes = 0;

    const uint64_t capacity = args.sendCfg.capacity > 0 ? static_cast<uint64_t>(args.sendCfg.capacity) : 1ull;
    while (consumedBytes < totalBytes) {
        uint64_t outstanding = nextProduce - nextConsume;
        if (producedBytes < totalBytes && outstanding < capacity) {
            const int32_t slotIndex = static_cast<int32_t>(nextProduce % capacity);
            const size_t payloadBytes =
                static_cast<size_t>(min_bytes(totalBytes - producedBytes, args.sendCfg.slotBytes));

            if (threadIdx.x == 0) {
                const uint64_t minSeq = (nextProduce >= capacity) ? nextProduce - capacity + 1 : 0ull;
                wait_for_credit(args.sendCfg, minSeq);
            }
            __syncthreads();

            SlotView outSlot = make_slot_view(args.sendCfg, slotIndex);
            copy_into_slot(
                outSlot, static_cast<const char*>(args.localInput) + args.sendOffset + producedBytes,
                static_cast<int32_t>(payloadBytes),
                [] __device__(char* dst, const void* src, int32_t bytes) { SlotOps<T>::copy(dst, src, bytes); });
            __syncthreads();

            if (threadIdx.x == 0) {
                publish_slot(outSlot, nextProduce);
            }
            __syncthreads();

            producedBytes += payloadBytes;
            nextProduce += 1;
            continue;
        }

        if (nextConsume < nextProduce) {
            const uint64_t recvCapacity =
                args.recvCfg.capacity > 0 ? static_cast<uint64_t>(args.recvCfg.capacity) : 1ull;
            const int32_t recvIndex = static_cast<int32_t>(nextConsume % recvCapacity);
            const size_t recvPayloadBytes =
                static_cast<size_t>(min_bytes(totalBytes - consumedBytes, args.recvCfg.slotBytes));

            if (threadIdx.x == 0) {
                SlotView waitSlot = make_slot_view(args.recvCfg, recvIndex);
                wait_for_slot(waitSlot, nextConsume);
            }
            __syncthreads();

            SlotView inSlot = make_slot_view(args.recvCfg, recvIndex);
            copy_from_slot(
                static_cast<char*>(args.localOutput) + args.recvOffset + consumedBytes, inSlot,
                static_cast<int32_t>(recvPayloadBytes),
                [] __device__(void* dst, const char* src, int32_t bytes) { SlotOps<T>::reduce(dst, src, bytes); });

            // CRITICAL: Ensure all threads have completed reading from the slot
            // BEFORE we update the gating sequence. Without this fence + sync,
            // there's a race window where the producer sees updated credit and
            // overwrites the slot while other threads are still reading.
            __threadfence();  // Ensure reads from slot are globally visible
            __syncthreads();  // All threads must complete before gating update

            if (threadIdx.x == 0) {
                store_release_u64(args.recvCfg.gating, nextConsume + 1);
            }
            __syncthreads();

            consumedBytes += recvPayloadBytes;
            nextConsume += 1;
            continue;
        }

        if (producedBytes >= totalBytes && nextConsume >= nextProduce) {
            break;
        }
    }

    __syncthreads();
    (void)producedBytes;
    (void)consumedBytes;
}

}  // namespace yali

#endif  // YALI_KERNELS_BANDWIDTH_CUH_
