/*************************************************************************
 * Yali AllReduce MPI - Simple API for Multi-Process
 *
 * ThunderKittens-style minimal interface for 2-process AllReduce.
 * Single header, zero boilerplate, full performance.
 *
 * Usage:
 *   #include "src/ops/allreduce_mpi.cuh"
 *
 *   // Setup (once at init) - initializes MPI and IPC
 *   yali::MPIComm comm(&argc, &argv);
 *
 *   // Allocate buffers on local GPU
 *   float *send, *recv;
 *   cudaMalloc(&send, count * sizeof(float));
 *   cudaMalloc(&recv, count * sizeof(float));
 *
 *   // AllReduce: each rank contributes send, all receive sum in recv
 *   yali::allreduce(comm, send, recv, count);
 *
 * NOTE: Unlike single-process API, each rank only manages its own buffers.
 * The API handles IPC exchange internally.
 *
 * Supports both kernel modes:
 * - Low-latency kernel: For messages <= 64MB (cp.async prefetch)
 * - Bandwidth kernel: For messages > 64MB (ring buffer, ~260 GB/s)
 *
 * See examples/02_multi_process/01_allreduce/simple_mpi.cu for usage.
 ************************************************************************/

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <vector>

#include "src/all_reduce/kernels.cuh"
#include "src/comm/comm.h"
#include "src/comm/ipc.cuh"
#include "src/include/yali_tuning.h"
#include "yali_launch.h"

// Bandwidth kernel entry point (defined in src/kernels/stream.cu)
extern "C" __global__ void _YaliKernel(YaliLaunchArgs args);

namespace yali {

// ============================================================================
// MPIComm: Manages multi-process communication context with IPC
// ============================================================================

// Maximum lanes we pre-allocate for (covers all practical cases)
static constexpr int kMPIMaxLanes = 128;

// Ring buffer state for stream kernel (per lane)
// Must match RingInfo struct layout in ipc.cu for IPC exchange compatibility
struct ManagedRing {
    uint64_t* sequence = nullptr;
    uint64_t* gating = nullptr;
    char* data = nullptr;
    int capacity = 0;
    size_t sequenceBytes = 0;
    size_t dataBytes = 0;
    // NOTE: No extra fields allowed - struct layout must match ipc.cu RingInfo exactly
};

// Separate sequence tracking (not part of IPC exchange)
struct RingSeqState {
    uint64_t seq_base = 0;  // Tracks sequence across calls
};

class MPIComm {
  public:
    YaliMPComm* comm_ = nullptr;
    bool initialized_ = false;
    bool ipc_exchanged_ = false;

    // Pre-allocated device args
    YaliLaunchArgs* args_dev_ = nullptr;

    // Host-side args for fast updates
    std::vector<YaliLaunchArgs> args_host_;

    // Cached peer pointer (after IPC exchange)
    void* peer_send_ = nullptr;
    void* cached_send_ptr_ = nullptr;  // For buffer_stable optimization

    // Bandwidth kernel state (lazy initialized)
    std::vector<ManagedRing> rings_;          // Local ring buffers (one per lane)
    std::vector<RingSeqState> ring_seq_;      // Sequence tracking (separate from IPC struct)
    std::vector<cudaStream_t> lane_streams_;  // Per-lane streams
    void** peer_sequence_ = nullptr;          // IPC pointers to peer's sequence arrays
    void** peer_gating_ = nullptr;            // IPC pointers to peer's gating
    void** peer_data_ = nullptr;              // IPC pointers to peer's data
    bool bw_initialized_ = false;
    int bw_lanes_ = 0;
    size_t bw_slot_bytes_ = 0;

    MPIComm() = default;

    // Initialize MPI communicator
    MPIComm(int* argc, char*** argv) {
        comm_ = yaliMPCommCreate(argc, argv);
        if (!comm_) {
            fprintf(stderr, "yali::MPIComm: Failed to create MPI communicator\n");
            return;
        }

        if (comm_->worldSize != 2) {
            fprintf(stderr, "yali::MPIComm: Requires exactly 2 ranks (got %d)\n", comm_->worldSize);
            yaliMPCommDestroy(comm_);
            comm_ = nullptr;
            return;
        }

        // Pre-allocate host args
        args_host_.resize(kMPIMaxLanes);

        // Pre-allocate device args on local GPU
        cudaError_t err = cudaMalloc(&args_dev_, kMPIMaxLanes * sizeof(YaliLaunchArgs));
        if (err != cudaSuccess) {
            fprintf(stderr, "yali::MPIComm: Failed to allocate device args\n");
            yaliMPCommDestroy(comm_);
            comm_ = nullptr;
            return;
        }

        initialized_ = true;
    }

    // No copy
    MPIComm(const MPIComm&) = delete;
    MPIComm& operator=(const MPIComm&) = delete;

    // Move constructor
    MPIComm(MPIComm&& other) noexcept
        : comm_(other.comm_), initialized_(other.initialized_), ipc_exchanged_(other.ipc_exchanged_),
          args_dev_(other.args_dev_), args_host_(std::move(other.args_host_)), peer_send_(other.peer_send_),
          cached_send_ptr_(other.cached_send_ptr_), rings_(std::move(other.rings_)),
          ring_seq_(std::move(other.ring_seq_)), lane_streams_(std::move(other.lane_streams_)),
          peer_sequence_(other.peer_sequence_), peer_gating_(other.peer_gating_), peer_data_(other.peer_data_),
          bw_initialized_(other.bw_initialized_), bw_lanes_(other.bw_lanes_), bw_slot_bytes_(other.bw_slot_bytes_) {
        other.comm_ = nullptr;
        other.args_dev_ = nullptr;
        other.peer_send_ = nullptr;
        other.cached_send_ptr_ = nullptr;
        other.peer_sequence_ = nullptr;
        other.peer_gating_ = nullptr;
        other.peer_data_ = nullptr;
        other.initialized_ = false;
        other.ipc_exchanged_ = false;
        other.bw_initialized_ = false;
    }

    // Move assignment
    MPIComm& operator=(MPIComm&& other) noexcept {
        if (this != &other) {
            cleanup();
            comm_ = other.comm_;
            initialized_ = other.initialized_;
            ipc_exchanged_ = other.ipc_exchanged_;
            args_dev_ = other.args_dev_;
            args_host_ = std::move(other.args_host_);
            peer_send_ = other.peer_send_;
            cached_send_ptr_ = other.cached_send_ptr_;
            rings_ = std::move(other.rings_);
            ring_seq_ = std::move(other.ring_seq_);
            lane_streams_ = std::move(other.lane_streams_);
            peer_sequence_ = other.peer_sequence_;
            peer_gating_ = other.peer_gating_;
            peer_data_ = other.peer_data_;
            bw_initialized_ = other.bw_initialized_;
            bw_lanes_ = other.bw_lanes_;
            bw_slot_bytes_ = other.bw_slot_bytes_;
            other.comm_ = nullptr;
            other.args_dev_ = nullptr;
            other.peer_send_ = nullptr;
            other.cached_send_ptr_ = nullptr;
            other.peer_sequence_ = nullptr;
            other.peer_gating_ = nullptr;
            other.peer_data_ = nullptr;
            other.initialized_ = false;
            other.ipc_exchanged_ = false;
            other.bw_initialized_ = false;
        }
        return *this;
    }

    ~MPIComm() { cleanup(); }

    void cleanup() {
        cleanup_bandwidth();

        if (args_dev_) {
            cudaFree(args_dev_);
            args_dev_ = nullptr;
        }
        if (comm_) {
            yaliMPCommDestroy(comm_);
            comm_ = nullptr;
        }
        initialized_ = false;
        ipc_exchanged_ = false;
    }

    void cleanup_bandwidth() {
        if (!bw_initialized_)
            return;

        // Free local ring buffers
        for (auto& ring : rings_) {
            if (ring.sequence)
                cudaFree(ring.sequence);
            if (ring.gating)
                cudaFree(ring.gating);
            if (ring.data)
                cudaFree(ring.data);
        }
        rings_.clear();
        ring_seq_.clear();

        // Destroy streams
        for (auto& s : lane_streams_) {
            if (s)
                cudaStreamDestroy(s);
        }
        lane_streams_.clear();

        // Free peer pointer arrays (allocated by yaliIpcExchangeRingBuffers)
        if (peer_sequence_) {
            free(peer_sequence_);
            peer_sequence_ = nullptr;
        }
        if (peer_gating_) {
            free(peer_gating_);
            peer_gating_ = nullptr;
        }
        if (peer_data_) {
            free(peer_data_);
            peer_data_ = nullptr;
        }

        bw_initialized_ = false;
        bw_lanes_ = 0;
    }

    bool ok() const { return initialized_ && comm_ && args_dev_; }

    int rank() const { return comm_ ? comm_->rank : -1; }
    int world_size() const { return comm_ ? comm_->worldSize : 0; }

    // Exchange IPC handles for a buffer
    // NOTE: IPC handle caching based on local pointer address is UNSAFE because:
    //   1. CUDA may reuse addresses after cudaFree
    //   2. The peer's buffer may have changed even if local pointer is the same
    //   3. Cross-process memory semantics require explicit re-exchange
    // When buffer_stable=true, user guarantees buffer won't be freed/reallocated.
    cudaError_t exchange_buffer(void* local_buf, size_t size, bool buffer_stable = false) {
        if (!ok())
            return cudaErrorInvalidValue;

        // If buffer_stable and pointer matches cached, skip re-exchange
        if (buffer_stable && ipc_exchanged_ && local_buf == cached_send_ptr_) {
            return cudaSuccess;
        }

        // Close previous IPC handles if any
        if (ipc_exchanged_) {
            yaliIpcCloseAll(comm_);
            ipc_exchanged_ = false;
        }

        // Exchange new handles
        int ret = yaliIpcExchangeBuffers(comm_, local_buf, size);
        if (ret != 0) {
            return cudaErrorUnknown;
        }

        // Get peer pointer
        int peer_rank = (comm_->rank + 1) % comm_->worldSize;
        peer_send_ = comm_->peerPtrs[peer_rank];
        ipc_exchanged_ = true;
        cached_send_ptr_ = local_buf;

        return cudaSuccess;
    }

    // Barrier synchronization
    void barrier() {
        if (comm_) {
            yaliMPCommBarrier(comm_);
        }
    }
};

// ============================================================================
// Internal: Kernel dispatch (users don't call this directly)
// ============================================================================

namespace detail {

// Helper to get NCCL datatype from C++ type
template <typename T>
struct NcclDtype {
    static constexpr ncclDataType_t value = ncclFloat;
};
template <>
struct NcclDtype<__half> {
    static constexpr ncclDataType_t value = ncclHalf;
};
template <>
struct NcclDtype<__nv_bfloat16> {
    static constexpr ncclDataType_t value = ncclBfloat16;
};

// Helper to get DType enum from C++ type (for heuristics)
template <typename T>
inline DType dtype_to_enum() {
    if (sizeof(T) == 4)
        return DType::FP32;
    // For 2-byte types, we can't distinguish FP16 from BF16, use FP16 as default
    return DType::FP16;
}

// Kernel config: auto-select based on message size
struct MPIKernelConfig {
    int lanes = 16;
    int ctas_per_lane = 4;
    int block_size = 512;
    bool use_flash = true;
    size_t smem_bytes = 0;
    size_t slot_bytes = 0;  // For stream kernel
};

template <typename T>
inline MPIKernelConfig mpi_auto_config(size_t count) {
    MPIKernelConfig cfg;
    size_t bytes = count * sizeof(T);
    DType dtype = dtype_to_enum<T>();

    // Use tuning heuristics - dtype-aware crossover selection
    size_t crossover = FlashCrossoverBytes(dtype);
    cfg.use_flash = (bytes <= crossover);

    if (cfg.use_flash) {
        cfg.lanes = FlashLanePreset(bytes, dtype);
        cfg.ctas_per_lane = 4;
        cfg.smem_bytes = FlashConfig::SharedMemoryBytes(cfg.block_size, 3, sizeof(T));
    } else {
        // Bandwidth kernel: always use 128 lanes for max throughput
        cfg.lanes = 128;
        cfg.block_size = 1024;
        cfg.ctas_per_lane = 1;
        cfg.slot_bytes = AutoSlotBytes(bytes);
        cfg.slot_bytes = ClampSlotBytes(cfg.slot_bytes, bytes);
    }
    return cfg;
}

// Setup YaliLaunchArgs for a single lane (MPI version)
inline void setup_mpi_lane_args(YaliLaunchArgs* args, void* local_in, void* local_out, void* peer_in, size_t count,
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

}  // namespace detail

// ============================================================================
// allreduce: Main MPI API
// ============================================================================

/**
 * AllReduce across 2 MPI ranks.
 *
 * Each rank provides its local send buffer and receives the sum in recv buffer.
 * IPC handles are exchanged automatically on first call (or when buffer changes).
 *
 * @param comm  Initialized MPIComm context
 * @param send  Local input buffer (on this rank's GPU)
 * @param recv  Local output buffer (on this rank's GPU)
 * @param count Number of elements per buffer
 * @param stream CUDA stream (default: 0)
 * @param buffer_stable If true, skip IPC re-exchange when send pointer matches
 *                      cached value. User guarantees buffer won't be freed/
 *                      reallocated between calls. Significantly improves perf.
 * @return cudaError_t
 */
template <typename T>
inline cudaError_t allreduce(MPIComm& comm, const T* send, T* recv, size_t count, cudaStream_t stream = 0,
                             bool buffer_stable = false) {
    if (!comm.ok())
        return cudaErrorInvalidValue;
    if (count == 0)
        return cudaSuccess;

    cudaError_t err;

    // Exchange IPC handles if needed (skip if buffer_stable and cached)
    err = comm.exchange_buffer(const_cast<T*>(send), count * sizeof(T), buffer_stable);
    if (err != cudaSuccess)
        return err;

    auto cfg = detail::mpi_auto_config<T>(count);

    const int rank = comm.rank();
    const int peer_rank = 1 - rank;

    if (!cfg.use_flash) {
        // =====================================================================
        // Bandwidth Kernel Path
        // =====================================================================

        // Lazy initialize bandwidth state if needed
        if (!comm.bw_initialized_ || comm.bw_lanes_ != cfg.lanes || comm.bw_slot_bytes_ != cfg.slot_bytes) {
            // Clean up previous state
            comm.cleanup_bandwidth();

            // Calculate per-lane allocation
            size_t base_elems = (count + cfg.lanes - 1) / cfg.lanes;

            // Allocate ring buffers for each lane
            comm.rings_.resize(cfg.lanes);
            comm.ring_seq_.resize(cfg.lanes);
            for (int lane = 0; lane < cfg.lanes; ++lane) {
                size_t start = static_cast<size_t>(lane) * base_elems;
                size_t end = (start + base_elems > count) ? count : start + base_elems;
                size_t lane_elems = (end > start) ? (end - start) : 0;
                size_t lane_bytes = lane_elems * sizeof(T);

                size_t capacity = (lane_bytes + cfg.slot_bytes - 1) / cfg.slot_bytes;
                if (capacity == 0)
                    capacity = 1;
                if (lane_bytes > 0 && capacity < 4)
                    capacity = 4;

                auto& ring = comm.rings_[lane];
                ring.capacity = static_cast<int>(capacity);
                ring.sequenceBytes = capacity * sizeof(uint64_t);
                ring.dataBytes = capacity * cfg.slot_bytes;
                comm.ring_seq_[lane].seq_base = 0;

                if (lane_elems == 0) {
                    ring.sequence = nullptr;
                    ring.gating = nullptr;
                    ring.data = nullptr;
                    continue;
                }

                // CUDA IPC requires allocations to be at least page-aligned
                // Pad small allocations for IPC compatibility (minimum ~512 bytes)
                size_t seq_alloc_bytes = ring.sequenceBytes < 512 ? 512 : ring.sequenceBytes;
                err = cudaMalloc(&ring.sequence, seq_alloc_bytes);
                if (err != cudaSuccess) {
                    comm.cleanup_bandwidth();
                    return err;
                }
                // Gating is only 8 bytes, but cudaMalloc pads to minimum
                err = cudaMalloc(&ring.gating, 512);
                if (err != cudaSuccess) {
                    comm.cleanup_bandwidth();
                    return err;
                }
                err = cudaMalloc(&ring.data, ring.dataBytes);
                if (err != cudaSuccess) {
                    comm.cleanup_bandwidth();
                    return err;
                }

                // Initialize sequence to -1 (0xff...) and gating to 0
                err = cudaMemset(ring.sequence, 0xff, ring.sequenceBytes);
                if (err != cudaSuccess) {
                    comm.cleanup_bandwidth();
                    return err;
                }
                err = cudaMemset(ring.gating, 0, sizeof(uint64_t));
                if (err != cudaSuccess) {
                    comm.cleanup_bandwidth();
                    return err;
                }
            }

            // Create per-lane streams
            comm.lane_streams_.resize(cfg.lanes);
            for (int lane = 0; lane < cfg.lanes; ++lane) {
                err = cudaStreamCreate(&comm.lane_streams_[lane]);
                if (err != cudaSuccess) {
                    comm.cleanup_bandwidth();
                    return err;
                }
            }

            // Sync before IPC exchange (required for cudaIpcGetMemHandle)
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                comm.cleanup_bandwidth();
                return err;
            }

            // Exchange ring buffer IPC handles
            // yaliIpcExchangeRingBuffers expects ManagedRing* (same layout as our struct)
            int ret = yaliIpcExchangeRingBuffers(comm.comm_, comm.rings_.data(), cfg.lanes, &comm.peer_sequence_,
                                                 &comm.peer_gating_, &comm.peer_data_);
            if (ret != 0) {
                comm.cleanup_bandwidth();
                return cudaErrorUnknown;
            }

            comm.bw_initialized_ = true;
            comm.bw_lanes_ = cfg.lanes;
            comm.bw_slot_bytes_ = cfg.slot_bytes;
        }

        // Calculate slots used for this message
        size_t base_elems = (count + cfg.lanes - 1) / cfg.lanes;
        std::vector<uint64_t> slots_used(cfg.lanes);
        for (int lane = 0; lane < cfg.lanes; ++lane) {
            size_t start = static_cast<size_t>(lane) * base_elems;
            size_t end = (start + base_elems > count) ? count : start + base_elems;
            size_t lane_elems = (end > start) ? (end - start) : 0;
            size_t lane_bytes = lane_elems * sizeof(T);
            slots_used[lane] = (lane_bytes == 0) ? 0 : (lane_bytes + cfg.slot_bytes - 1) / cfg.slot_bytes;
        }

        // Setup args for each lane
        const int slot_bytes_int = static_cast<int>(cfg.slot_bytes);
        for (int lane = 0; lane < cfg.lanes; ++lane) {
            size_t start = static_cast<size_t>(lane) * base_elems;
            size_t end = (start + base_elems > count) ? count : start + base_elems;
            size_t lane_elems = (end > start) ? (end - start) : 0;
            size_t offset_bytes = start * sizeof(T);

            auto& args = comm.args_host_[lane];
            args = {};

            // Point to lane's portion of the buffers
            args.localInput = reinterpret_cast<char*>(const_cast<T*>(send)) + offset_bytes;
            args.localOutput = reinterpret_cast<char*>(recv) + offset_bytes;
            args.peerInput = reinterpret_cast<char*>(comm.peer_send_) + offset_bytes;
            args.elementCount = lane_elems;
            args.elementSize = sizeof(T);
            args.sendOffset = 0;  // Already offset
            args.recvOffset = 0;
            args.datatype = detail::NcclDtype<T>::value;
            args.redOp = ncclSum;
            args.rank = rank;
            args.worldSize = 2;
            args.laneIndex = lane;
            args.laneCount = cfg.lanes;
            args.ctasPerLane = cfg.ctas_per_lane;
            args.flash = 0;

            // Setup ring buffer pointers
            if (lane_elems > 0) {
                int peer_idx = peer_rank * cfg.lanes + lane;

                // Send to peer's ring buffers
                args.sendSequence = reinterpret_cast<uint64_t*>(comm.peer_sequence_[peer_idx]);
                args.sendGating = reinterpret_cast<uint64_t*>(comm.peer_gating_[peer_idx]);
                args.sendData = reinterpret_cast<char*>(comm.peer_data_[peer_idx]);
                args.sendCapacity = comm.rings_[lane].capacity;
                args.sendSlotBytes = slot_bytes_int;
                args.sendSlotStride = slot_bytes_int;

                // Receive from our ring buffers
                args.recvSequence = comm.rings_[lane].sequence;
                args.recvGating = comm.rings_[lane].gating;
                args.recvData = comm.rings_[lane].data;
                args.recvCapacity = comm.rings_[lane].capacity;
                args.recvSlotBytes = slot_bytes_int;
                args.recvSlotStride = slot_bytes_int;

                args.initialSequence = comm.ring_seq_[lane].seq_base;
            }
        }

        // Barrier before kernel launch
        comm.barrier();

        // Launch stream kernel on per-lane streams
        dim3 bw_grid(1);
        dim3 bw_block(cfg.block_size);
        for (int lane = 0; lane < cfg.lanes; ++lane) {
            if (comm.rings_[lane].sequence == nullptr)
                continue;

            // Launch kernel using cudaLaunchKernel (matches raw harness)
            void* kernelArgs[] = {&comm.args_host_[lane]};
            err =
                cudaLaunchKernel((const void*)_YaliKernel, bw_grid, bw_block, kernelArgs, 0, comm.lane_streams_[lane]);
            if (err != cudaSuccess)
                return err;
        }

        // Wait for all lanes to complete
        for (int lane = 0; lane < cfg.lanes; ++lane) {
            err = cudaStreamSynchronize(comm.lane_streams_[lane]);
            if (err != cudaSuccess)
                return err;
        }

        // Update sequence bases for next call
        for (int lane = 0; lane < cfg.lanes; ++lane) {
            comm.ring_seq_[lane].seq_base += slots_used[lane];
        }

        // Barrier after kernel completion
        comm.barrier();

        return cudaSuccess;
    }

    // =========================================================================
    // Low-Latency Kernel Path
    // =========================================================================

    // Update host args
    for (int lane = 0; lane < cfg.lanes; ++lane) {
        detail::setup_mpi_lane_args(&comm.args_host_[lane], const_cast<T*>(send), recv, comm.peer_send_, count,
                                    sizeof(T), rank, lane, cfg.lanes, cfg.ctas_per_lane);
    }

    // Copy args to device (async)
    err = cudaMemcpyAsync(comm.args_dev_, comm.args_host_.data(), cfg.lanes * sizeof(YaliLaunchArgs),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
        return err;

    // Set shared mem attribute
    err = cudaFuncSetAttribute((const void*)FlashKernel<T, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(cfg.smem_bytes));
    if (err != cudaSuccess)
        return err;

    // Barrier before kernel launch
    comm.barrier();

    // Launch kernel
    dim3 grid(cfg.lanes * cfg.ctas_per_lane);
    dim3 block(cfg.block_size);

    FlashKernel<T, 3><<<grid, block, cfg.smem_bytes, stream>>>(comm.args_dev_, cfg.lanes, cfg.ctas_per_lane);
    err = cudaGetLastError();
    if (err != cudaSuccess)
        return err;

    // Sync and barrier after kernel
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
        return err;

    comm.barrier();

    return cudaSuccess;
}

}  // namespace yali
