/*************************************************************************
 * Copyright (c) 2025
 * All rights reserved.
 *
 * Ring buffer primitives for Yali inter-GPU communication.
 ************************************************************************/

#ifndef YALI_KERNELS_RING_CUH_
#define YALI_KERNELS_RING_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdint.h>

namespace yali {

// Device-side helpers for acquire/release semantics. On Volta and newer
// architectures we can emit explicit acquire/release operations; otherwise we
// fall back to a system fence to keep compilation working across all SMs.
//
// Memory ordering notes:
// - store_release: ensures all prior stores are visible before this store
// - load_acquire: ensures subsequent loads see values from stores that happened-before
//
// Implementation:
// - __threadfence_system() ensures visibility across all GPU/CPU address spaces
// - For release: fence BEFORE store ensures prior writes are flushed
// - For acquire: fence AFTER load ensures dependent reads see prior writes
//
// Note: In strict C++11 semantics, acquire fence would be __threadfence_system()
// BEFORE dependent operations. However, in GPU memory model with volatile loads,
// the fence after load ensures the L2 cache is synchronized before proceeding.
// This pattern matches CUDA documentation for multi-GPU memory ordering.

__device__ inline void store_release_u64(uint64_t* addr, uint64_t value) {
    // Release semantics: ensure all prior writes are visible before this store
    __threadfence_system();
    *reinterpret_cast<volatile uint64_t*>(addr) = value;
}

__device__ inline uint64_t load_acquire_u64(const uint64_t* addr) {
    // Acquire semantics: volatile load followed by fence to ensure
    // subsequent dependent operations see values from prior stores.
    // The fence synchronizes the L2 cache view after the load.
    uint64_t value = *reinterpret_cast<const volatile uint64_t*>(addr);
    __threadfence_system();
    return value;
}

// Adaptive busy-spin with nanosleep backoff that mirrors the lock-free ring buffer
// strategy. Keeps latency low while avoiding monopolising the SM.
//
// IPC Latency Considerations:
// - Intra-process NVLink: ~100-500ns typical
// - Cross-process IPC via CUDA: ~1-10µs typical
// - PCIe peer access: ~5-20µs typical
//
// The backoff starts aggressive (100ns) and scales up to 1µs for sustained spins.
// This balances latency sensitivity with power efficiency.
__device__ inline void busy_spin_with_hint(int spin) {
#if __CUDA_ARCH__ >= 700
    if (spin < 16) {
        __nanosleep(100);  // First 16 spins: 100ns (fast path for NVLink)
    } else if (spin < 64) {
        __nanosleep(256);  // Next 48 spins: 256ns
    } else if (spin < 256) {
        __nanosleep(512);  // Next 192 spins: 512ns
    } else {
        __nanosleep(1000);  // Long waits: 1µs (IPC/PCIe scale)
    }
#else
    (void)spin;  // Fallback: no-op on older architectures.
#endif
}

// Shared ring configuration held in NVLink-visible global memory.
struct RingConfig {
    int32_t capacity;    // number of slots (expect power-of-two)
    int32_t slotBytes;   // payload bytes per slot
    int32_t slotStride;  // stride between slots (>= slotBytes)
    uint64_t* sequence;  // producer sequence values, one per slot
    uint64_t* gating;    // consumer gating sequence (single entry)
    char* data;          // base pointer to payload buffer
};

struct alignas(16) SlotView {
    char* ptr;
    uint64_t* seqAddr;
    int32_t bytes;
    int32_t stride;
};

__device__ inline SlotView make_slot_view(const RingConfig& cfg, int32_t slot) {
    SlotView view;
    view.bytes = cfg.slotBytes;
    view.stride = cfg.slotStride;
    view.ptr = cfg.data + static_cast<size_t>(slot) * cfg.slotStride;
    view.seqAddr = cfg.sequence + slot;
    return view;
}

__device__ inline uint64_t wait_for_credit(const RingConfig& cfg, uint64_t minSequence) {
    int spin = 0;
    while (true) {
        uint64_t gate = load_acquire_u64(cfg.gating);
        if (gate >= minSequence)
            return gate;
        busy_spin_with_hint(spin++);
    }
}

__device__ inline void publish_slot(SlotView slot, uint64_t sequence) {
    // Note: store_release_u64 already includes the fence before the store,
    // so we don't need an additional fence here. The sequence was previously
    // double-fencing which is harmless but wasteful.
    store_release_u64(slot.seqAddr, sequence);
}

__device__ inline void wait_for_slot(const SlotView& slot, uint64_t expectedSequence) {
    int spin = 0;
    while (true) {
        uint64_t seq = load_acquire_u64(slot.seqAddr);
        if (seq == expectedSequence)
            break;
        busy_spin_with_hint(spin++);
    }
}

template <typename CopyFn>
__device__ inline void copy_into_slot(const SlotView& slot, const void* src, int32_t bytes, CopyFn&& copy) {
    copy(slot.ptr, src, bytes);
}

template <typename CopyFn>
__device__ inline void copy_from_slot(void* dst, const SlotView& slot, int32_t bytes, CopyFn&& copy) {
    copy(dst, slot.ptr, bytes);
}

}  // namespace yali

#endif  // YALI_KERNELS_RING_CUH_
