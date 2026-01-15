/*************************************************************************
 * Yali AllReduce - Simple API
 *
 * ThunderKittens-style minimal interface for 2-GPU AllReduce.
 * Single header, zero boilerplate, full performance.
 *
 * Usage:
 *   #include "src/ops/allreduce.cuh"
 *
 *   // Setup (once at init) - pre-allocates device memory
 *   yali::Comm comm(0, 1);
 *
 *   // AllReduce: reads from send buffers, writes sum to recv buffers
 *   yali::allreduce(comm, send0, recv0, send1, recv1, count);
 *
 * NOTE: This API uses separate send/recv buffers (like NCCL). The kernel
 * reads from send buffers and writes the reduced result to recv buffers.
 * send and recv may alias if you want in-place semantics.
 *
 * Supports both kernel modes:
 * - Low-latency kernel: For messages <= 64MB (cp.async prefetch)
 * - Bandwidth kernel: For messages > 64MB (ring buffer, ~260 GB/s)
 *
 * See examples/simple_all_reduce.cu for a complete example.
 ************************************************************************/

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "src/all_reduce/kernels.cuh"
#include "src/common/peer_access.cuh"
#include "src/include/yali_tuning.h"
#include "yali_launch.h"

// Bandwidth kernel entry point (defined in src/kernels/stream.cu)
extern "C" __global__ void _YaliKernel(YaliLaunchArgs args);

namespace yali {

// ============================================================================
// Profiling support (opt-in via YALI_PROFILE_KERNELS=1)
// ============================================================================
namespace detail {
inline bool profile_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = getenv("YALI_PROFILE_KERNELS");
        enabled = (env && atoi(env) > 0) ? 1 : 0;
    }
    return enabled > 0;
}
}  // namespace detail

// ============================================================================
// Comm: Manages 2-GPU communication context with pre-allocated args
// ============================================================================

// Maximum lanes we pre-allocate for (covers all practical cases)
static constexpr int kMaxLanes = 128;

// Ring buffer state for stream kernel (per lane, per GPU)
struct RingState {
    uint64_t* sequence = nullptr;  // Sequence array (one per slot)
    uint64_t* gating = nullptr;    // Consumer gating sequence (single entry)
    char* data = nullptr;          // Payload data buffer
    int capacity = 0;              // Number of slots
    uint64_t seq_base = 0;         // Sequence base (tracks across calls)
};

struct Comm {
    int gpu0, gpu1;
    bool initialized = false;

    // Pre-allocated device args (one array per GPU, sized for max lanes)
    YaliLaunchArgs* args0_dev = nullptr;
    YaliLaunchArgs* args1_dev = nullptr;

    // Host-side args for fast updates
    std::vector<YaliLaunchArgs> args0_host;
    std::vector<YaliLaunchArgs> args1_host;

    // Bandwidth kernel ring buffer state (lazy initialized)
    std::vector<RingState> ring0_;             // Ring buffers for GPU0 (one per lane)
    std::vector<RingState> ring1_;             // Ring buffers for GPU1 (one per lane)
    std::vector<cudaStream_t> streams0_;       // Per-lane streams for GPU0
    std::vector<cudaStream_t> streams1_;       // Per-lane streams for GPU1
    std::vector<cudaEvent_t> start0_, stop0_;  // Events for GPU0
    std::vector<cudaEvent_t> start1_, stop1_;  // Events for GPU1
    bool bw_initialized_ = false;
    int bw_lanes_ = 0;
    size_t bw_slot_bytes_ = 0;

    // Call counter for fire-and-forget mode (sequence base pre-computation)
    uint64_t call_count_ = 0;
    size_t last_count_ = 0;                 // Last element count (for stable buffer optimization)
    std::vector<uint64_t> slots_per_lane_;  // Cached slots per lane

    Comm() = default;

    // Initialize with two GPU indices - pre-allocates device memory
    Comm(int g0, int g1) : gpu0(g0), gpu1(g1) {
        if (!EnableBidirectionalP2P(g0, g1)) {
            fprintf(stderr, "yali::Comm: Failed to enable P2P between GPU %d and %d\n", g0, g1);
            return;
        }

        // Pre-allocate host args
        args0_host.resize(kMaxLanes);
        args1_host.resize(kMaxLanes);

        // Pre-allocate device args on GPU0
        cudaError_t err = cudaSetDevice(g0);
        if (err != cudaSuccess)
            return;
        err = cudaMalloc(&args0_dev, kMaxLanes * sizeof(YaliLaunchArgs));
        if (err != cudaSuccess)
            return;

        // Pre-allocate device args on GPU1
        err = cudaSetDevice(g1);
        if (err != cudaSuccess) {
            cudaSetDevice(g0);
            cudaFree(args0_dev);
            args0_dev = nullptr;
            return;
        }
        err = cudaMalloc(&args1_dev, kMaxLanes * sizeof(YaliLaunchArgs));
        if (err != cudaSuccess) {
            cudaSetDevice(g0);
            cudaFree(args0_dev);
            args0_dev = nullptr;
            return;
        }

        initialized = true;
    }

    // Move constructor
    Comm(Comm&& other) noexcept
        : gpu0(other.gpu0), gpu1(other.gpu1), initialized(other.initialized), args0_dev(other.args0_dev),
          args1_dev(other.args1_dev), args0_host(std::move(other.args0_host)), args1_host(std::move(other.args1_host)),
          ring0_(std::move(other.ring0_)), ring1_(std::move(other.ring1_)), streams0_(std::move(other.streams0_)),
          streams1_(std::move(other.streams1_)), start0_(std::move(other.start0_)), stop0_(std::move(other.stop0_)),
          start1_(std::move(other.start1_)), stop1_(std::move(other.stop1_)), bw_initialized_(other.bw_initialized_),
          bw_lanes_(other.bw_lanes_), bw_slot_bytes_(other.bw_slot_bytes_), call_count_(other.call_count_),
          last_count_(other.last_count_), slots_per_lane_(std::move(other.slots_per_lane_)) {
        other.args0_dev = nullptr;
        other.args1_dev = nullptr;
        other.initialized = false;
        other.bw_initialized_ = false;
        other.call_count_ = 0;
    }

    // Move assignment
    Comm& operator=(Comm&& other) noexcept {
        if (this != &other) {
            cleanup();
            gpu0 = other.gpu0;
            gpu1 = other.gpu1;
            initialized = other.initialized;
            args0_dev = other.args0_dev;
            args1_dev = other.args1_dev;
            args0_host = std::move(other.args0_host);
            args1_host = std::move(other.args1_host);
            ring0_ = std::move(other.ring0_);
            ring1_ = std::move(other.ring1_);
            streams0_ = std::move(other.streams0_);
            streams1_ = std::move(other.streams1_);
            start0_ = std::move(other.start0_);
            stop0_ = std::move(other.stop0_);
            start1_ = std::move(other.start1_);
            stop1_ = std::move(other.stop1_);
            bw_initialized_ = other.bw_initialized_;
            bw_lanes_ = other.bw_lanes_;
            bw_slot_bytes_ = other.bw_slot_bytes_;
            call_count_ = other.call_count_;
            last_count_ = other.last_count_;
            slots_per_lane_ = std::move(other.slots_per_lane_);
            other.args0_dev = nullptr;
            other.args1_dev = nullptr;
            other.initialized = false;
            other.bw_initialized_ = false;
            other.call_count_ = 0;
        }
        return *this;
    }

    // No copy
    Comm(const Comm&) = delete;
    Comm& operator=(const Comm&) = delete;

    ~Comm() { cleanup(); }

    void cleanup() {
        // Cleanup stream kernel state
        cleanup_bandwidth();

        if (args0_dev) {
            cudaSetDevice(gpu0);
            cudaFree(args0_dev);
            args0_dev = nullptr;
        }
        if (args1_dev) {
            cudaSetDevice(gpu1);
            cudaFree(args1_dev);
            args1_dev = nullptr;
        }
        initialized = false;
    }

    void cleanup_bandwidth() {
        if (!bw_initialized_)
            return;

        // Cleanup GPU0 ring buffers
        cudaSetDevice(gpu0);
        for (auto& ring : ring0_) {
            if (ring.sequence)
                cudaFree(ring.sequence);
            if (ring.gating)
                cudaFree(ring.gating);
            if (ring.data)
                cudaFree(ring.data);
        }
        for (auto& s : streams0_)
            if (s)
                cudaStreamDestroy(s);
        for (auto& e : start0_)
            if (e)
                cudaEventDestroy(e);
        for (auto& e : stop0_)
            if (e)
                cudaEventDestroy(e);

        // Cleanup GPU1 ring buffers
        cudaSetDevice(gpu1);
        for (auto& ring : ring1_) {
            if (ring.sequence)
                cudaFree(ring.sequence);
            if (ring.gating)
                cudaFree(ring.gating);
            if (ring.data)
                cudaFree(ring.data);
        }
        for (auto& s : streams1_)
            if (s)
                cudaStreamDestroy(s);
        for (auto& e : start1_)
            if (e)
                cudaEventDestroy(e);
        for (auto& e : stop1_)
            if (e)
                cudaEventDestroy(e);

        ring0_.clear();
        ring1_.clear();
        streams0_.clear();
        streams1_.clear();
        start0_.clear();
        stop0_.clear();
        start1_.clear();
        stop1_.clear();
        bw_initialized_ = false;
        bw_lanes_ = 0;
    }

    bool ok() const { return initialized && args0_dev && args1_dev; }

    // Synchronize all pending operations (call after fire-and-forget mode)
    cudaError_t sync() {
        cudaError_t err;

        // Sync flash kernel streams (uses default stream)
        err = cudaSetDevice(gpu0);
        if (err != cudaSuccess)
            return err;
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            return err;

        err = cudaSetDevice(gpu1);
        if (err != cudaSuccess)
            return err;
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            return err;

        // Sync stream kernel streams (per-lane)
        if (bw_initialized_) {
            err = cudaSetDevice(gpu0);
            if (err != cudaSuccess)
                return err;
            for (int lane = 0; lane < bw_lanes_; ++lane) {
                if (ring0_[lane].capacity > 0) {
                    err = cudaStreamSynchronize(streams0_[lane]);
                    if (err != cudaSuccess)
                        return err;
                }
            }

            err = cudaSetDevice(gpu1);
            if (err != cudaSuccess)
                return err;
            for (int lane = 0; lane < bw_lanes_; ++lane) {
                if (ring1_[lane].capacity > 0) {
                    err = cudaStreamSynchronize(streams1_[lane]);
                    if (err != cudaSuccess)
                        return err;
                }
            }
        }

        return cudaSuccess;
    }
};

// ============================================================================
// Internal: Kernel dispatch (users don't call this directly)
// ============================================================================

namespace detail {

// Kernel config: auto-select based on message size
struct KernelConfig {
    int lanes = 16;
    int ctas_per_lane = 4;
    int block_size = 512;
    bool use_flash = true;
    size_t smem_bytes = 0;
    size_t slot_bytes = 0;  // For stream kernel
};

template <typename T>
inline DType dtype_to_enum() {
    if (sizeof(T) == 4)
        return DType::FP32;
    // For 2-byte types, we can't distinguish FP16 from BF16, use FP16 as default
    return DType::FP16;
}

template <typename T>
inline KernelConfig auto_config(size_t count) {
    KernelConfig cfg;
    size_t bytes = count * sizeof(T);
    DType dtype = dtype_to_enum<T>();

    // Use tuning heuristics from yali_tuning.h
    size_t crossover = FlashCrossoverBytes(dtype);
    cfg.use_flash = (bytes <= crossover);

    if (cfg.use_flash) {
        cfg.lanes = FlashLanePreset(bytes, dtype);
        cfg.block_size = 512;
        // Compute CTAs per lane based on tile size
        int vector_elems = 16 / static_cast<int>(sizeof(T));
        size_t tile_elems = static_cast<size_t>(cfg.block_size) * 3 * vector_elems;
        size_t base_lane_elems = (count + cfg.lanes - 1) / cfg.lanes;
        cfg.ctas_per_lane = AutoCtasPerLane(true, cfg.lanes, base_lane_elems, tile_elems);
        cfg.smem_bytes = FlashConfig::SharedMemoryBytes(cfg.block_size, 3, sizeof(T));
    } else {
        // Bandwidth kernel: always use 128 lanes for max throughput
        // (matches raw harness behavior from sweep tuning)
        cfg.lanes = 128;
        cfg.block_size = 1024;  // Bandwidth kernel uses 1024 threads
        cfg.ctas_per_lane = 1;  // One CTA per lane for stream kernel
        cfg.slot_bytes = AutoSlotBytes(bytes);
        cfg.slot_bytes = ClampSlotBytes(cfg.slot_bytes, bytes);
    }

    // Clamp lanes to max
    if (cfg.lanes > kMaxLanes)
        cfg.lanes = kMaxLanes;
    if (cfg.lanes < 1)
        cfg.lanes = 1;

    return cfg;
}

// Initialize stream kernel ring buffers for a GPU
inline cudaError_t init_ring_for_gpu(int device, std::vector<RingState>& rings, std::vector<cudaStream_t>& streams,
                                     std::vector<cudaEvent_t>& starts, std::vector<cudaEvent_t>& stops, int lanes,
                                     size_t count, size_t elem_size, size_t slot_bytes) {
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess)
        return err;

    rings.resize(lanes);
    streams.resize(lanes);
    starts.resize(lanes);
    stops.resize(lanes);

    size_t base_lane_elems = (count + lanes - 1) / lanes;

    for (int lane = 0; lane < lanes; ++lane) {
        size_t start_elem = static_cast<size_t>(lane) * base_lane_elems;
        size_t end_elem = (start_elem + base_lane_elems > count) ? count : start_elem + base_lane_elems;
        size_t lane_elems = (end_elem > start_elem) ? (end_elem - start_elem) : 0;
        size_t lane_bytes = lane_elems * elem_size;

        // Create stream and events
        err = cudaStreamCreate(&streams[lane]);
        if (err != cudaSuccess)
            return err;
        err = cudaEventCreate(&starts[lane]);
        if (err != cudaSuccess)
            return err;
        err = cudaEventCreate(&stops[lane]);
        if (err != cudaSuccess)
            return err;

        if (lane_elems == 0) {
            rings[lane] = {};
            continue;
        }

        // Calculate capacity (number of slots)
        size_t capacity = (lane_bytes + slot_bytes - 1) / slot_bytes;
        if (capacity == 0)
            capacity = 1;
        if (capacity < 4)
            capacity = 4;  // Minimum for ring buffer efficiency

        rings[lane].capacity = static_cast<int>(capacity);

        // Allocate sequence array
        err = cudaMalloc(&rings[lane].sequence, capacity * sizeof(uint64_t));
        if (err != cudaSuccess)
            return err;
        err = cudaMemset(rings[lane].sequence, 0xff, capacity * sizeof(uint64_t));
        if (err != cudaSuccess)
            return err;

        // Allocate gating
        err = cudaMalloc(&rings[lane].gating, sizeof(uint64_t));
        if (err != cudaSuccess)
            return err;
        err = cudaMemset(rings[lane].gating, 0, sizeof(uint64_t));
        if (err != cudaSuccess)
            return err;

        // Allocate data
        err = cudaMalloc(&rings[lane].data, capacity * slot_bytes);
        if (err != cudaSuccess)
            return err;

        rings[lane].seq_base = 0;
    }

    return cudaSuccess;
}

// Setup YaliLaunchArgs for a single lane
inline void setup_lane_args(YaliLaunchArgs* args, void* local_in, void* local_out, void* peer_in, size_t count,
                            size_t elem_size, int rank, int lane, int lane_count, int ctas_per_lane) {
    size_t base_elems = (count + lane_count - 1) / lane_count;
    size_t start = static_cast<size_t>(lane) * base_elems;
    size_t end = (start + base_elems > count) ? count : start + base_elems;
    size_t elems = (end > start) ? (end - start) : 0;
    size_t offset = start * elem_size;

    *args = {};
    args->localInput = local_in;
    args->localOutput = local_out;
    args->peerInput = peer_in;
    args->elementCount = elems;
    args->elementSize = elem_size;
    args->sendOffset = offset;
    args->recvOffset = offset;
    args->rank = rank;
    args->laneIndex = lane;
    args->laneCount = lane_count;
    args->ctasPerLane = ctas_per_lane;
    args->flash = 1;
}

// Setup stream kernel args for a single lane
inline void setup_bw_lane_args(YaliLaunchArgs* args, void* local_in, void* local_out, void* peer_in,
                               const RingState& send_ring, const RingState& recv_ring, size_t count, size_t elem_size,
                               size_t slot_bytes, int rank, int lane, int lane_count) {
    size_t base_elems = (count + lane_count - 1) / lane_count;
    size_t start = static_cast<size_t>(lane) * base_elems;
    size_t end = (start + base_elems > count) ? count : start + base_elems;
    size_t elems = (end > start) ? (end - start) : 0;
    size_t offset = start * elem_size;

    *args = {};

    // Ring buffer config - send to peer's ring, receive from our ring
    args->sendSequence = send_ring.sequence;
    args->sendGating = send_ring.gating;
    args->sendData = send_ring.data;
    args->sendCapacity = send_ring.capacity;
    args->sendSlotBytes = static_cast<int32_t>(slot_bytes);
    args->sendSlotStride = static_cast<int32_t>(slot_bytes);

    args->recvSequence = recv_ring.sequence;
    args->recvGating = recv_ring.gating;
    args->recvData = recv_ring.data;
    args->recvCapacity = recv_ring.capacity;
    args->recvSlotBytes = static_cast<int32_t>(slot_bytes);
    args->recvSlotStride = static_cast<int32_t>(slot_bytes);

    // Local buffers - point to lane's portion (offset applied directly)
    args->localInput = static_cast<char*>(local_in) + offset;
    args->localOutput = static_cast<char*>(local_out) + offset;
    args->peerInput = static_cast<char*>(peer_in) + offset;
    args->elementCount = elems;
    args->elementSize = elem_size;
    args->sendOffset = 0;  // Already offset above
    args->recvOffset = 0;
    args->initialSequence = recv_ring.seq_base;

    args->datatype = ncclFloat;  // Will be overridden based on T
    args->redOp = ncclSum;
    args->rank = rank;
    args->worldSize = 2;
    args->laneIndex = lane;
    args->laneCount = lane_count;
    args->ctasPerLane = 1;
    args->flash = 0;
}

// Calculate slots used for a lane
inline uint64_t calc_slots_used(size_t lane_bytes, size_t slot_bytes) {
    if (lane_bytes == 0 || slot_bytes == 0)
        return 0;
    return (lane_bytes + slot_bytes - 1) / slot_bytes;
}

}  // namespace detail

// ============================================================================
// allreduce: Main API
// ============================================================================

/**
 * AllReduce across 2 GPUs.
 *
 * Reads from send buffers on each GPU, computes element-wise sum,
 * writes result to recv buffers on both GPUs.
 *
 * Automatically selects kernel based on message size:
 * - Low-latency kernel for <= 64MB (cp.async prefetch)
 * - Bandwidth kernel for > 64MB (ring buffer, ~260 GB/s)
 *
 * @param comm  Initialized Comm context
 * @param send0, recv0  Input/output buffers on GPU0
 * @param send1, recv1  Input/output buffers on GPU1
 * @param count Number of elements per buffer
 * @param stream CUDA stream (default: 0, ignored for stream kernel which uses internal streams)
 * @param sync  If true (default), synchronize after the call. If false, fire-and-forget mode
 *              for maximum throughput. Call comm.sync() when you need to wait.
 * @return cudaError_t
 */
template <typename T>
inline cudaError_t allreduce(Comm& comm, const T* send0, T* recv0, const T* send1, T* recv1, size_t count,
                             cudaStream_t stream = 0, bool sync = true) {
    if (!comm.ok())
        return cudaErrorInvalidValue;
    if (count == 0)
        return cudaSuccess;

    auto cfg = detail::auto_config<T>(count);
    cudaError_t err;

    // ========================================================================
    // Bandwidth kernel path for large messages
    // ========================================================================
    if (!cfg.use_flash) {
        // Lazy initialize ring buffers if needed
        bool reinit_needed = !comm.bw_initialized_ || comm.bw_lanes_ != cfg.lanes || comm.last_count_ != count;
        if (reinit_needed) {
            comm.cleanup_bandwidth();

            err = detail::init_ring_for_gpu(comm.gpu0, comm.ring0_, comm.streams0_, comm.start0_, comm.stop0_,
                                            cfg.lanes, count, sizeof(T), cfg.slot_bytes);
            if (err != cudaSuccess)
                return err;

            err = detail::init_ring_for_gpu(comm.gpu1, comm.ring1_, comm.streams1_, comm.start1_, comm.stop1_,
                                            cfg.lanes, count, sizeof(T), cfg.slot_bytes);
            if (err != cudaSuccess)
                return err;

            comm.bw_initialized_ = true;
            comm.bw_lanes_ = cfg.lanes;
            comm.bw_slot_bytes_ = cfg.slot_bytes;
            comm.last_count_ = count;
            comm.call_count_ = 0;  // Reset on reinit

            // Pre-compute slots per lane (cached for fire-and-forget)
            comm.slots_per_lane_.resize(cfg.lanes);
            size_t base_lane_elems = (count + cfg.lanes - 1) / cfg.lanes;
            for (int lane = 0; lane < cfg.lanes; ++lane) {
                size_t start_elem = static_cast<size_t>(lane) * base_lane_elems;
                size_t end_elem = (start_elem + base_lane_elems > count) ? count : start_elem + base_lane_elems;
                size_t lane_elems = (end_elem > start_elem) ? (end_elem - start_elem) : 0;
                size_t lane_bytes = lane_elems * sizeof(T);
                comm.slots_per_lane_[lane] = detail::calc_slots_used(lane_bytes, cfg.slot_bytes);
            }
        }

        size_t base_lane_elems = (count + cfg.lanes - 1) / cfg.lanes;
        const dim3 grid(1);
        const dim3 block(cfg.block_size);

        // Pre-compute sequence base for this iteration (fire-and-forget friendly)
        uint64_t iter_idx = comm.call_count_;

        // Launch stream kernel for each lane on both GPUs
        for (int lane = 0; lane < cfg.lanes; ++lane) {
            size_t start_elem = static_cast<size_t>(lane) * base_lane_elems;
            size_t end_elem = (start_elem + base_lane_elems > count) ? count : start_elem + base_lane_elems;
            size_t lane_elems = (end_elem > start_elem) ? (end_elem - start_elem) : 0;

            if (lane_elems == 0)
                continue;

            // Compute sequence base from call count (not ring state)
            uint64_t lane_seq_base = iter_idx * comm.slots_per_lane_[lane];
            comm.ring0_[lane].seq_base = lane_seq_base;
            comm.ring1_[lane].seq_base = lane_seq_base;

            // GPU0: sends to GPU1's ring, receives from own ring
            detail::setup_bw_lane_args(&comm.args0_host[lane], const_cast<T*>(send0), recv0, const_cast<T*>(send1),
                                       comm.ring1_[lane], comm.ring0_[lane], count, sizeof(T), cfg.slot_bytes, 0, lane,
                                       cfg.lanes);

            // GPU1: sends to GPU0's ring, receives from own ring
            detail::setup_bw_lane_args(&comm.args1_host[lane], const_cast<T*>(send1), recv1, const_cast<T*>(send0),
                                       comm.ring0_[lane], comm.ring1_[lane], count, sizeof(T), cfg.slot_bytes, 1, lane,
                                       cfg.lanes);

            // Launch on GPU0 (with optional profiling events)
            err = cudaSetDevice(comm.gpu0);
            if (err != cudaSuccess)
                return err;

            if (detail::profile_enabled()) {
                err = cudaEventRecord(comm.start0_[lane], comm.streams0_[lane]);
                if (err != cudaSuccess)
                    return err;
            }
            void* kernel_args0[] = {&comm.args0_host[lane]};
            err = cudaLaunchKernel((const void*)_YaliKernel, grid, block, kernel_args0, 0, comm.streams0_[lane]);
            if (err != cudaSuccess)
                return err;
            if (detail::profile_enabled()) {
                err = cudaEventRecord(comm.stop0_[lane], comm.streams0_[lane]);
                if (err != cudaSuccess)
                    return err;
            }

            // Launch on GPU1 (with optional profiling events)
            err = cudaSetDevice(comm.gpu1);
            if (err != cudaSuccess)
                return err;

            if (detail::profile_enabled()) {
                err = cudaEventRecord(comm.start1_[lane], comm.streams1_[lane]);
                if (err != cudaSuccess)
                    return err;
            }
            void* kernel_args1[] = {&comm.args1_host[lane]};
            err = cudaLaunchKernel((const void*)_YaliKernel, grid, block, kernel_args1, 0, comm.streams1_[lane]);
            if (err != cudaSuccess)
                return err;
            if (detail::profile_enabled()) {
                err = cudaEventRecord(comm.stop1_[lane], comm.streams1_[lane]);
                if (err != cudaSuccess)
                    return err;
            }
        }

        // Increment call count (for fire-and-forget sequence tracking)
        comm.call_count_++;

        // Conditional sync (skip for fire-and-forget mode)
        if (sync) {
            err = cudaSetDevice(comm.gpu0);
            if (err != cudaSuccess)
                return err;
            for (int lane = 0; lane < cfg.lanes; ++lane) {
                if (comm.ring0_[lane].capacity > 0) {
                    err = cudaStreamSynchronize(comm.streams0_[lane]);
                    if (err != cudaSuccess)
                        return err;
                }
            }

            err = cudaSetDevice(comm.gpu1);
            if (err != cudaSuccess)
                return err;
            for (int lane = 0; lane < cfg.lanes; ++lane) {
                if (comm.ring1_[lane].capacity > 0) {
                    err = cudaStreamSynchronize(comm.streams1_[lane]);
                    if (err != cudaSuccess)
                        return err;
                }
            }

            // Print kernel profiling info if enabled via YALI_PROFILE_KERNELS=1
            if (detail::profile_enabled()) {
                float maxMs = 0.0f;
                for (int lane = 0; lane < cfg.lanes; ++lane) {
                    if (comm.ring0_[lane].capacity == 0)
                        continue;
                    float ms0 = 0.0f, ms1 = 0.0f;
                    cudaSetDevice(comm.gpu0);
                    cudaEventElapsedTime(&ms0, comm.start0_[lane], comm.stop0_[lane]);
                    cudaSetDevice(comm.gpu1);
                    cudaEventElapsedTime(&ms1, comm.start1_[lane], comm.stop1_[lane]);
                    if (ms0 > maxMs)
                        maxMs = ms0;
                    if (ms1 > maxMs)
                        maxMs = ms1;
                }
                printf("[YALI_PROFILE] BW kernel: max_lane_ms=%.3f lanes=%d\n", maxMs, cfg.lanes);
            }
        }

        return cudaSuccess;
    }

    // ========================================================================
    // Low-latency kernel path for small messages
    // ========================================================================

    // Update host args for GPU0
    for (int lane = 0; lane < cfg.lanes; ++lane) {
        detail::setup_lane_args(&comm.args0_host[lane], const_cast<T*>(send0), recv0, const_cast<T*>(send1), count,
                                sizeof(T), 0, lane, cfg.lanes, cfg.ctas_per_lane);
    }

    // Update host args for GPU1
    for (int lane = 0; lane < cfg.lanes; ++lane) {
        detail::setup_lane_args(&comm.args1_host[lane], const_cast<T*>(send1), recv1, const_cast<T*>(send0), count,
                                sizeof(T), 1, lane, cfg.lanes, cfg.ctas_per_lane);
    }

    // Copy args to device (async)
    err = cudaSetDevice(comm.gpu0);
    if (err != cudaSuccess)
        return err;
    err = cudaMemcpyAsync(comm.args0_dev, comm.args0_host.data(), cfg.lanes * sizeof(YaliLaunchArgs),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
        return err;

    err = cudaSetDevice(comm.gpu1);
    if (err != cudaSuccess)
        return err;
    err = cudaMemcpyAsync(comm.args1_dev, comm.args1_host.data(), cfg.lanes * sizeof(YaliLaunchArgs),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
        return err;

    // Set shared mem attribute (only needs to be done once, but cheap)
    err = cudaFuncSetAttribute((const void*)FlashKernel<T, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(cfg.smem_bytes));
    if (err != cudaSuccess)
        return err;

    // Launch on both GPUs
    dim3 grid(cfg.lanes * cfg.ctas_per_lane);
    dim3 block(cfg.block_size);

    err = cudaSetDevice(comm.gpu0);
    if (err != cudaSuccess)
        return err;
    FlashKernel<T, 3><<<grid, block, cfg.smem_bytes, stream>>>(comm.args0_dev, cfg.lanes, cfg.ctas_per_lane);
    err = cudaGetLastError();
    if (err != cudaSuccess)
        return err;

    err = cudaSetDevice(comm.gpu1);
    if (err != cudaSuccess)
        return err;
    FlashKernel<T, 3><<<grid, block, cfg.smem_bytes, stream>>>(comm.args1_dev, cfg.lanes, cfg.ctas_per_lane);
    err = cudaGetLastError();
    if (err != cudaSuccess)
        return err;

    // Conditional sync (skip for fire-and-forget mode)
    if (sync) {
        err = cudaSetDevice(comm.gpu0);
        if (err != cudaSuccess)
            return err;
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
            return err;

        err = cudaSetDevice(comm.gpu1);
        if (err != cudaSuccess)
            return err;
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
            return err;
    }

    return cudaSuccess;
}

}  // namespace yali
