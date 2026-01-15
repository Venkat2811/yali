/**
 * YALI AllReduce Benchmark (benchmark_yali)
 *
 * This benchmark mimics how inference engines actually use AllReduce:
 * 1. Setup communicators/buffers once
 * 2. Run N allreduce calls in a tight loop (no sync between)
 * 3. Sync only at end
 * 4. Measure total throughput
 *
 * Supports both Flash mode (small messages) and Stream mode (large messages).
 * Automatically switches based on yali::FlashCrossoverBytes().
 *
 * Multi-dtype support: fp32, fp16, bf16 (set via YALI_DTYPE env or --dtype arg)
 *
 * This is a fair comparison to benchmark_nccl.cu (NCCL version).
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// Public headers from src/include/
#include "yali_launch.h"
#include "yali_tuning.h"

// AllReduce kernels
#include "src/all_reduce/kernels.cuh"

// Common utilities
#include "src/common/buffer_ops.cuh"
#include "src/common/hw_info.cuh"
#include "src/common/peer_access.cuh"
#include "src/common/validation.cuh"

// Stream kernel entry point
extern "C" __global__ void _YaliKernel(YaliLaunchArgs args);

#define CHECK_CUDA(cmd)                                                                                                \
    do {                                                                                                               \
        cudaError_t e = cmd;                                                                                           \
        if (e != cudaSuccess) {                                                                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                      \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

//------------------------------------------------------------------------------
// Data Type Configuration (multi-dtype support: fp32, fp16, bf16)
//------------------------------------------------------------------------------
enum class HarnessDTypeKind {
    kFloat32 = 0,
    kFloat16 = 1,
    kBFloat16 = 2,
};

struct HarnessDTypeConfig {
    HarnessDTypeKind kind;
    ncclDataType_t ncclType;
    size_t elementSize;
    const char* name;
    yali::DType tuningDtype;  // For lane/crossover heuristics
};

static HarnessDTypeConfig ParseDType(const char* dtypeStr) {
    std::string lowered = dtypeStr ? std::string(dtypeStr) : std::string("f32");
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lowered == "f16" || lowered == "fp16" || lowered == "float16") {
        return {HarnessDTypeKind::kFloat16, ncclHalf, sizeof(__half), "fp16", yali::DType::FP16};
    }
    if (lowered == "bf16" || lowered == "bfloat16") {
        return {HarnessDTypeKind::kBFloat16, ncclBfloat16, sizeof(__nv_bfloat16), "bf16", yali::DType::BF16};
    }
    return {HarnessDTypeKind::kFloat32, ncclFloat, sizeof(float), "fp32", yali::DType::FP32};
}

static HarnessDTypeConfig GetDTypeFromEnv() {
    const char* env = std::getenv("YALI_DTYPE");
    return ParseDType(env);
}

// Ring buffer for Stream kernel
struct ManagedRing {
    uint64_t* sequence = nullptr;
    uint64_t* gating = nullptr;
    char* data = nullptr;
    int capacity = 0;
    size_t sequenceBytes = 0;
    size_t dataBytes = 0;
};

enum class KernelMode { Auto, Flash, Stream };

//------------------------------------------------------------------------------
// Timing Mode for benchmarks (ThunderKittens-compatible)
//------------------------------------------------------------------------------
enum class TimingMode {
    Throughput,  // Wall-clock, fire-and-forget, single sync at end
    Latency,     // Wall-clock, sync after each iteration
    CudaEvents   // CUDA events around batch (matches ThunderKittens exactly)
};

// Helper to get timing mode name
static const char* TimingModeName(TimingMode mode) {
    switch (mode) {
        case TimingMode::Throughput:
            return "THROUGHPUT (wall-clock)";
        case TimingMode::Latency:
            return "LATENCY (wall-clock)";
        case TimingMode::CudaEvents:
            return "CUDA_EVENTS (GPU-only, ThunderKittens-style)";
        default:
            return "UNKNOWN";
    }
}

//------------------------------------------------------------------------------
// Flash Mode Benchmark (templated for multi-dtype support)
//------------------------------------------------------------------------------
template <typename T>
void benchmarkFlashTyped(size_t elemCount, int numCalls, int warmupCalls, bool verify, const HarnessDTypeConfig& dtype,
                         int lanesOverride = 0, TimingMode timingMode = TimingMode::Throughput) {
    constexpr int kRanks = 2;
    const size_t bytes = elemCount * dtype.elementSize;

    // Flash kernel config
    const int blockSize = 512;
    const int prefetchStages = 3;
    const size_t sharedBytes = static_cast<size_t>(blockSize * prefetchStages * 16);

    // Use auto-tuned lane count (dtype-aware) or override
    int lanes = (lanesOverride > 0) ? lanesOverride : yali::FlashLanePreset(bytes, dtype.tuningDtype);
    if (lanes < 1) lanes = 1;
    if (lanes > 128) lanes = 128;

    // Calculate CTAs per lane
    const int vectorElems = 16 / dtype.elementSize;
    const size_t tileElems = static_cast<size_t>(blockSize * prefetchStages * vectorElems);
    const size_t baseLaneElems = (elemCount + lanes - 1) / lanes;
    const int ctasPerLane = yali::AutoCtasPerLane(true, lanes, baseLaneElems, tileElems);

    // Enable peer access
    yali::EnablePeerAccessOrDie(0, 1);
    yali::EnablePeerAccessOrDie(1, 0);

    // Allocate buffers
    T* send[kRanks];
    T* recv[kRanks];
    cudaStream_t streams[kRanks];

    for (int r = 0; r < kRanks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaMalloc(&send[r], bytes));
        CHECK_CUDA(cudaMalloc(&recv[r], bytes));
        CHECK_CUDA(cudaStreamCreate(&streams[r]));
        CHECK_CUDA(yali::SeedBufferSync(send[r], elemCount, static_cast<T>(r + 1)));
        CHECK_CUDA(cudaMemset(recv[r], 0, bytes));
    }

    // Set shared memory attribute
    for (int r = 0; r < kRanks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaFuncSetAttribute((const void*)yali::FlashKernel<T, 3>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sharedBytes)));
    }

    // Setup launch args for each lane
    std::vector<std::vector<YaliLaunchArgs>> hostArgs(kRanks, std::vector<YaliLaunchArgs>(lanes));
    std::vector<YaliLaunchArgs*> deviceArgs(kRanks);

    for (int r = 0; r < kRanks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaMalloc(&deviceArgs[r], lanes * sizeof(YaliLaunchArgs)));

        for (int lane = 0; lane < lanes; lane++) {
            size_t startElem = static_cast<size_t>(lane) * baseLaneElems;
            size_t endElem = std::min(startElem + baseLaneElems, elemCount);
            size_t laneElems = (endElem > startElem) ? (endElem - startElem) : 0;
            size_t offsetBytes = startElem * dtype.elementSize;

            auto& args = hostArgs[r][lane];
            args = {};
            args.localInput = send[r];
            args.localOutput = recv[r];
            args.peerInput = send[(r + 1) % kRanks];
            args.elementCount = laneElems;
            args.elementSize = dtype.elementSize;
            args.sendOffset = offsetBytes;
            args.recvOffset = offsetBytes;
            args.datatype = dtype.ncclType;
            args.redOp = ncclSum;
            args.rank = r;
            args.laneIndex = lane;
            args.laneCount = lanes;
            args.ctasPerLane = ctasPerLane;
            args.flash = 1;
        }
        CHECK_CUDA(
            cudaMemcpy(deviceArgs[r], hostArgs[r].data(), lanes * sizeof(YaliLaunchArgs), cudaMemcpyHostToDevice));
    }

    const dim3 grid(lanes * ctasPerLane);
    const dim3 block(blockSize);

    printf("Mode: FLASH | lanes=%d, ctasPerLane=%d, grid=%d, block=%d\n", lanes, ctasPerLane, grid.x, block.x);
    printf("Timing mode: %s\n", TimingModeName(timingMode));

    // Lambda for launching one iteration
    auto launchIteration = [&]() {
        for (int r = 0; r < kRanks; r++) {
            CHECK_CUDA(cudaSetDevice(r));
            yali::FlashKernel<T, 3><<<grid, block, sharedBytes, streams[r]>>>(deviceArgs[r], lanes, ctasPerLane);
        }
    };

    // Sync all helper
    auto syncAll = [&]() {
        for (int r = 0; r < kRanks; r++) {
            CHECK_CUDA(cudaSetDevice(r));
            CHECK_CUDA(cudaStreamSynchronize(streams[r]));
        }
    };

    // Warmup
    for (int iter = 0; iter < warmupCalls; iter++) {
        launchIteration();
    }
    syncAll();

    // Timed run - depends on timing mode
    double totalMs = 0.0;

    if (timingMode == TimingMode::CudaEvents) {
        // CUDA events around batch (ThunderKittens methodology)
        cudaEvent_t startEvent, stopEvent;
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaEventCreate(&startEvent));
        CHECK_CUDA(cudaEventCreate(&stopEvent));

        // Pre-barrier to ensure GPU is idle
        syncAll();

        // Record start on stream 0
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaEventRecord(startEvent, streams[0]));

        // Fire all iterations
        for (int iter = 0; iter < numCalls; iter++) {
            launchIteration();
        }

        // Record stop on stream 0 and sync
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaEventRecord(stopEvent, streams[0]));
        syncAll();

        float elapsedMs = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
        totalMs = static_cast<double>(elapsedMs);

        CHECK_CUDA(cudaEventDestroy(startEvent));
        CHECK_CUDA(cudaEventDestroy(stopEvent));

    } else if (timingMode == TimingMode::Throughput) {
        // Wall-clock, fire-and-forget, single sync at end
        auto start = std::chrono::steady_clock::now();

        for (int iter = 0; iter < numCalls; iter++) {
            launchIteration();
        }
        syncAll();

        auto end = std::chrono::steady_clock::now();
        totalMs = std::chrono::duration<double, std::milli>(end - start).count();

    } else {
        // Latency mode: sync after each iteration
        auto start = std::chrono::steady_clock::now();

        for (int iter = 0; iter < numCalls; iter++) {
            launchIteration();
            syncAll();
        }

        auto end = std::chrono::steady_clock::now();
        totalMs = std::chrono::duration<double, std::milli>(end - start).count();
    }

    double avgUs = (totalMs * 1000.0) / numCalls;

    // NCCL busBw formula for AllReduce: data_size * 2 * (nranks-1) / nranks / time
    // For 2 GPUs: factor = 2 * (2-1) / 2 = 1.0, so busBw = data_size / time
    constexpr int nranks = kRanks;
    double dataBytes = static_cast<double>(bytes);
    double busBwFactor = 2.0 * static_cast<double>(nranks - 1) / static_cast<double>(nranks);
    double gbps = (dataBytes * busBwFactor / 1e9) / (avgUs / 1e6);
    double solPercent = gbps / 100.0 * 100.0;  // vs 100 GB/s unidirectional NVLink

    const char* modeStr = (timingMode == TimingMode::CudaEvents)   ? "cuda-events"
                          : (timingMode == TimingMode::Throughput) ? "throughput"
                                                                   : "latency";
    printf("YALI (Flash-%s, %s): %d calls, %.2f us/call avg, %.2f GB/s (%.1f%% SoL)\n", dtype.name, modeStr, numCalls,
           avgUs, gbps, solPercent);

    // Correctness verification
    if (verify) {
        bool allOk = true;
        for (int r = 0; r < kRanks; r++) {
            CHECK_CUDA(cudaSetDevice(r));
            bool rankOk = yali::ValidateRankResult(recv[r], elemCount, r, kRanks);
            if (!rankOk) {
                printf("FAILED: Rank %d validation failed\n", r);
                allOk = false;
            }
        }
        printf("Correctness: %s\n", allOk ? "PASSED" : "FAILED");
    }

    // Cleanup
    for (int r = 0; r < kRanks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        cudaFree(deviceArgs[r]);
        cudaFree(send[r]);
        cudaFree(recv[r]);
        cudaStreamDestroy(streams[r]);
    }
}

// Dispatch function for Flash benchmark (selects template based on dtype)
void benchmarkFlash(size_t elemCount, int numCalls, int warmupCalls, bool verify, const HarnessDTypeConfig& dtype,
                    int lanesOverride = 0, TimingMode timingMode = TimingMode::Throughput) {
    switch (dtype.kind) {
        case HarnessDTypeKind::kFloat16:
            benchmarkFlashTyped<__half>(elemCount, numCalls, warmupCalls, verify, dtype, lanesOverride, timingMode);
            break;
        case HarnessDTypeKind::kBFloat16:
            benchmarkFlashTyped<__nv_bfloat16>(elemCount, numCalls, warmupCalls, verify, dtype, lanesOverride, timingMode);
            break;
        case HarnessDTypeKind::kFloat32:
        default:
            benchmarkFlashTyped<float>(elemCount, numCalls, warmupCalls, verify, dtype, lanesOverride, timingMode);
            break;
    }
}

//------------------------------------------------------------------------------
// Stream Mode Benchmark - Production-like implementation (templated)
//
// Key insight: We can PRE-COMPUTE sequence bases for all iterations,
// then launch without per-iteration sync. The ring buffer's gating
// mechanism handles flow control on the GPU side.
//------------------------------------------------------------------------------
template <typename T>
void benchmarkStreamTyped(size_t elemCount, int numCalls, int warmupCalls, bool verify, const HarnessDTypeConfig& dtype,
                          int lanesOverride = 0, TimingMode timingMode = TimingMode::Throughput) {
    constexpr int kRanks = 2;
    const size_t bytes = elemCount * dtype.elementSize;

    // Stream kernel config - allow override for testing (dtype-aware)
    int lanes = (lanesOverride > 0) ? lanesOverride : yali::StreamLanePreset(bytes, dtype.tuningDtype);
    if (lanes < 1)
        lanes = 1;
    if (lanes > 128)
        lanes = 128;
    const int blockSize = 1024;
    const int ctasPerLane = 1;

    // Ring buffer slot sizing
    size_t ringSlotBytes = yali::AutoSlotBytes(bytes);
    ringSlotBytes = yali::ClampSlotBytes(ringSlotBytes, bytes);
    const int ringSlotBytesInt = static_cast<int>(ringSlotBytes);

    // Enable peer access
    yali::EnablePeerAccessOrDie(0, 1);
    yali::EnablePeerAccessOrDie(1, 0);

    // Allocate send/recv buffers
    T* send[kRanks];
    T* recv[kRanks];
    // Stream mode uses per-lane streams for parallel execution across lanes
    std::vector<std::vector<cudaStream_t>> laneStreams(kRanks, std::vector<cudaStream_t>(lanes));

    for (int r = 0; r < kRanks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaMalloc(&send[r], bytes));
        CHECK_CUDA(cudaMalloc(&recv[r], bytes));
        for (int lane = 0; lane < lanes; lane++) {
            CHECK_CUDA(cudaStreamCreate(&laneStreams[r][lane]));
        }
        CHECK_CUDA(yali::SeedBufferSync(send[r], elemCount, static_cast<T>(r + 1)));
        CHECK_CUDA(cudaMemset(recv[r], 0, bytes));
    }

    // Calculate lane distribution
    const size_t baseLaneElems = (elemCount + lanes - 1) / lanes;
    std::vector<size_t> laneOffsets(lanes);
    std::vector<size_t> laneElements(lanes);
    std::vector<uint64_t> laneSlotsUsed(lanes, 0);

    for (int lane = 0; lane < lanes; lane++) {
        size_t startElem = static_cast<size_t>(lane) * baseLaneElems;
        size_t endElem = std::min(startElem + baseLaneElems, elemCount);
        laneOffsets[lane] = startElem;
        laneElements[lane] = (endElem > startElem) ? (endElem - startElem) : 0;

        size_t laneBytes = laneElements[lane] * dtype.elementSize;
        laneSlotsUsed[lane] = (laneBytes == 0) ? 0 : (laneBytes + ringSlotBytes - 1) / ringSlotBytes;
    }

    // Allocate ring buffers for each rank and lane
    std::vector<std::vector<ManagedRing>> laneRing(kRanks, std::vector<ManagedRing>(lanes));

    for (int r = 0; r < kRanks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        for (int lane = 0; lane < lanes; lane++) {
            size_t laneElems = laneElements[lane];
            size_t laneBytes = laneElems * dtype.elementSize;

            size_t laneCapacity = (laneBytes + ringSlotBytes - 1) / ringSlotBytes;
            if (laneCapacity == 0)
                laneCapacity = 1;
            if (laneBytes > 0 && laneCapacity < 4)
                laneCapacity = 4;

            laneRing[r][lane].capacity = static_cast<int>(laneCapacity);
            laneRing[r][lane].sequenceBytes = laneCapacity * sizeof(uint64_t);
            laneRing[r][lane].dataBytes = laneCapacity * ringSlotBytes;

            if (laneElems == 0) {
                laneRing[r][lane].sequence = nullptr;
                laneRing[r][lane].gating = nullptr;
                laneRing[r][lane].data = nullptr;
                continue;
            }

            CHECK_CUDA(cudaMalloc(&laneRing[r][lane].sequence, laneRing[r][lane].sequenceBytes));
            CHECK_CUDA(cudaMalloc(&laneRing[r][lane].gating, sizeof(uint64_t)));
            CHECK_CUDA(cudaMalloc(&laneRing[r][lane].data, laneRing[r][lane].dataBytes));
            CHECK_CUDA(cudaMemset(laneRing[r][lane].sequence, 0xff, laneRing[r][lane].sequenceBytes));
            CHECK_CUDA(cudaMemset(laneRing[r][lane].gating, 0, sizeof(uint64_t)));
        }
    }

    // Setup BASE launch args for each rank and lane (will update initialSequence per iteration)
    std::vector<std::vector<YaliLaunchArgs>> launchArgs(kRanks, std::vector<YaliLaunchArgs>(lanes));

    for (int r = 0; r < kRanks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        for (int lane = 0; lane < lanes; lane++) {
            size_t elems = laneElements[lane];
            size_t offsetElems = laneOffsets[lane];

            auto& args = launchArgs[r][lane];
            args = {};

            // Send to peer's ring buffers
            args.sendSequence = laneRing[(r + 1) % kRanks][lane].sequence;
            args.sendGating = laneRing[(r + 1) % kRanks][lane].gating;
            args.sendData = laneRing[(r + 1) % kRanks][lane].data;
            args.sendCapacity = laneRing[(r + 1) % kRanks][lane].capacity;
            args.sendSlotBytes = ringSlotBytesInt;
            args.sendSlotStride = ringSlotBytesInt;

            // Receive from own ring buffers
            args.recvSequence = laneRing[r][lane].sequence;
            args.recvGating = laneRing[r][lane].gating;
            args.recvData = laneRing[r][lane].data;
            args.recvCapacity = laneRing[r][lane].capacity;
            args.recvSlotBytes = ringSlotBytesInt;
            args.recvSlotStride = ringSlotBytesInt;

            args.localInput = reinterpret_cast<char*>(send[r]) + offsetElems * dtype.elementSize;
            args.localOutput = reinterpret_cast<char*>(recv[r]) + offsetElems * dtype.elementSize;
            args.peerInput = reinterpret_cast<char*>(send[(r + 1) % kRanks]) + offsetElems * dtype.elementSize;
            args.elementCount = elems;
            args.elementSize = dtype.elementSize;
            args.sendOffset = 0;
            args.recvOffset = 0;
            args.initialSequence = 0;  // Will be set per-iteration
            args.datatype = dtype.ncclType;
            args.redOp = ncclSum;
            args.rank = r;
            args.laneIndex = lane;
            args.laneCount = lanes;
            args.ctasPerLane = ctasPerLane;
            args.flash = 0;
        }
    }

    const dim3 grid(1);
    const dim3 block(blockSize);

    // Calculate total kernel launches per iteration for visibility
    int kernelsPerIter = 0;
    for (int lane = 0; lane < lanes; lane++) {
        if (laneElements[lane] > 0)
            kernelsPerIter += kRanks;
    }

    printf("================================================================================\n");
    printf("YALI Stream Mode Benchmark - Production-like Implementation (%s)\n", dtype.name);
    printf("================================================================================\n");
    printf("Config:\n");
    printf("  Data type:      %s\n", dtype.name);
    printf("  Data size:      %.2f MB (%zu elements)\n", bytes / 1e6, elemCount);
    printf("  Lanes:          %d\n", lanes);
    printf("  Slot size:      %zu bytes\n", ringSlotBytes);
    printf("  Block size:     %d threads\n", blockSize);
    printf("  Kernels/iter:   %d (lanes x ranks)\n", kernelsPerIter);
    printf("  Timing mode:    %s\n", TimingModeName(timingMode));
    printf("  Warmup:         %d iterations\n", warmupCalls);
    printf("  Measured:       %d iterations\n", numCalls);
    printf("--------------------------------------------------------------------------------\n");

    // Track sequence base across all iterations (warmup + measured)
    uint64_t globalIterCount = 0;

    // Lambda to launch one iteration (no sync, just launch)
    auto launchIteration = [&](uint64_t iterIdx) {
        // Pre-compute sequence base for this iteration
        // This is the KEY insight: we can compute this without waiting for GPU
        for (int r = 0; r < kRanks; r++) {
            for (int lane = 0; lane < lanes; lane++) {
                launchArgs[r][lane].initialSequence = iterIdx * laneSlotsUsed[lane];
            }
        }

        // Launch kernels for all ranks and lanes
        for (int r = 0; r < kRanks; r++) {
            CHECK_CUDA(cudaSetDevice(r));
            for (int lane = 0; lane < lanes; lane++) {
                if (laneElements[lane] == 0)
                    continue;
                void* kernelParams[] = {&launchArgs[r][lane]};
                CHECK_CUDA(
                    cudaLaunchKernel((const void*)_YaliKernel, grid, block, kernelParams, 0, laneStreams[r][lane]));
            }
        }
    };

    // Lambda to sync all streams
    auto syncAll = [&]() {
        for (int r = 0; r < kRanks; r++) {
            CHECK_CUDA(cudaSetDevice(r));
            for (int lane = 0; lane < lanes; lane++) {
                if (laneElements[lane] == 0)
                    continue;
                CHECK_CUDA(cudaStreamSynchronize(laneStreams[r][lane]));
            }
        }
    };

    // ==========================================================================
    // WARMUP PHASE (always with sync to ensure correctness)
    // ==========================================================================
    printf("Running warmup...\n");
    for (int iter = 0; iter < warmupCalls; iter++) {
        launchIteration(globalIterCount++);
        syncAll();  // Warmup always syncs to ensure stable state
    }
    printf("Warmup complete.\n");

    // ==========================================================================
    // TIMED RUN
    // ==========================================================================
    printf("Running timed iterations...\n");

    double totalMs = 0.0;

    if (timingMode == TimingMode::CudaEvents) {
        // CUDA EVENTS MODE: Matches ThunderKittens exactly
        // Records GPU timestamps around the batch, excludes host overhead
        //
        // This is the "GPU Speed-of-Light" measurement that measures
        // only kernel execution time, not launch overhead.

        // Create events on GPU0 (will capture when all work completes)
        cudaEvent_t startEvent, stopEvent;
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaEventCreate(&startEvent));
        CHECK_CUDA(cudaEventCreate(&stopEvent));

        // Pre-barrier to ensure clean start
        for (int r = 0; r < kRanks; r++) {
            CHECK_CUDA(cudaSetDevice(r));
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // Record start event on GPU0's first lane stream
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaEventRecord(startEvent, laneStreams[0][0]));

        // Fire all iterations without waiting
        for (int iter = 0; iter < numCalls; iter++) {
            launchIteration(globalIterCount++);
        }

        // Record stop event on GPU0's first lane stream
        // (will wait for all prior work on this stream)
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaEventRecord(stopEvent, laneStreams[0][0]));

        // Sync all streams to ensure completion
        syncAll();

        // Get elapsed time from GPU events
        float elapsedMs = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
        totalMs = static_cast<double>(elapsedMs);

        // Cleanup events
        CHECK_CUDA(cudaEventDestroy(startEvent));
        CHECK_CUDA(cudaEventDestroy(stopEvent));

    } else if (timingMode == TimingMode::Throughput) {
        // THROUGHPUT MODE: Wall-clock, fire-and-forget, single sync at end
        // This measures total time including launch overhead (what inference engines see)

        // Pre-barrier to ensure clean start
        for (int r = 0; r < kRanks; r++) {
            CHECK_CUDA(cudaSetDevice(r));
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        auto start = std::chrono::steady_clock::now();

        // Fire all iterations without waiting
        for (int iter = 0; iter < numCalls; iter++) {
            launchIteration(globalIterCount++);
            // NO SYNC - fire and forget!
        }

        // Single sync at end
        syncAll();

        auto end = std::chrono::steady_clock::now();
        totalMs = std::chrono::duration<double, std::milli>(end - start).count();

    } else {
        // LATENCY MODE: Wall-clock, sync after each iteration
        // This measures end-to-end including driver overhead (BS=1 scenario)

        auto start = std::chrono::steady_clock::now();

        for (int iter = 0; iter < numCalls; iter++) {
            launchIteration(globalIterCount++);
            syncAll();  // Wait for completion
        }

        auto end = std::chrono::steady_clock::now();
        totalMs = std::chrono::duration<double, std::milli>(end - start).count();
    }

    // ==========================================================================
    // RESULTS
    // ==========================================================================
    double avgUs = (totalMs * 1000.0) / numCalls;

    // NCCL busBw formula for AllReduce: data_size * 2 * (nranks-1) / nranks / time
    // For 2 GPUs: factor = 2 * (2-1) / 2 = 1.0, so busBw = data_size / time
    constexpr int nranks = kRanks;
    double dataBytes = static_cast<double>(bytes);
    double busBwFactor = 2.0 * static_cast<double>(nranks - 1) / static_cast<double>(nranks);
    double gbps = (dataBytes * busBwFactor / 1e9) / (avgUs / 1e6);

    // Calculate Speed-of-Light (assuming NV4 = 100 GB/s unidirectional)
    double nvlinkUniGBs = 100.0;  // A100 NV4 unidirectional
    double solPercent = (gbps / nvlinkUniGBs) * 100.0;

    printf("--------------------------------------------------------------------------------\n");
    printf("Results:\n");
    printf("  Total time:     %.2f ms\n", totalMs);
    printf("  Avg latency:    %.2f us/call\n", avgUs);
    printf("  Bus bandwidth:  %.2f GB/s\n", gbps);
    printf("  Speed-of-Light: %.1f%% (of %.0f GB/s NVLink uni)\n", solPercent, nvlinkUniGBs);
    printf("--------------------------------------------------------------------------------\n");

    // One-line summary for easy parsing
    const char* modeStr = (timingMode == TimingMode::CudaEvents)   ? "cuda-events"
                          : (timingMode == TimingMode::Throughput) ? "throughput"
                                                                   : "latency";
    printf("YALI (Stream-%s, %s): %d calls, %.2f us/call, %.2f GB/s, %.1f%% SoL\n", dtype.name, modeStr, numCalls,
           avgUs, gbps, solPercent);

    // ==========================================================================
    // CORRECTNESS VERIFICATION
    // ==========================================================================
    if (verify) {
        printf("\nVerifying correctness...\n");
        bool allOk = true;
        for (int r = 0; r < kRanks; r++) {
            CHECK_CUDA(cudaSetDevice(r));
            bool rankOk = yali::ValidateRankResult(recv[r], elemCount, r, kRanks);
            if (!rankOk) {
                printf("  FAILED: Rank %d validation failed\n", r);
                allOk = false;
            } else {
                printf("  Rank %d: PASSED\n", r);
            }
        }
        printf("Correctness: %s\n", allOk ? "PASSED" : "FAILED");
    }

    printf("================================================================================\n");

    // Cleanup
    for (int r = 0; r < kRanks; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        for (int lane = 0; lane < lanes; lane++) {
            if (laneRing[r][lane].sequence)
                cudaFree(laneRing[r][lane].sequence);
            if (laneRing[r][lane].gating)
                cudaFree(laneRing[r][lane].gating);
            if (laneRing[r][lane].data)
                cudaFree(laneRing[r][lane].data);
            cudaStreamDestroy(laneStreams[r][lane]);
        }
        cudaFree(send[r]);
        cudaFree(recv[r]);
    }
}

// Dispatch function for Stream benchmark (selects template based on dtype)
void benchmarkStream(size_t elemCount, int numCalls, int warmupCalls, bool verify, const HarnessDTypeConfig& dtype,
                     int lanesOverride = 0, TimingMode timingMode = TimingMode::Throughput) {
    switch (dtype.kind) {
        case HarnessDTypeKind::kFloat16:
            benchmarkStreamTyped<__half>(elemCount, numCalls, warmupCalls, verify, dtype, lanesOverride, timingMode);
            break;
        case HarnessDTypeKind::kBFloat16:
            benchmarkStreamTyped<__nv_bfloat16>(elemCount, numCalls, warmupCalls, verify, dtype, lanesOverride,
                                                timingMode);
            break;
        case HarnessDTypeKind::kFloat32:
        default:
            benchmarkStreamTyped<float>(elemCount, numCalls, warmupCalls, verify, dtype, lanesOverride, timingMode);
            break;
    }
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    size_t elemCount = 262144;  // 1MB (default)
    int numCalls = 1000;
    int warmupCalls = 100;
    bool verify = false;
    KernelMode mode = KernelMode::Auto;
    int lanesOverride = 0;                           // 0 = use auto
    TimingMode timingMode = TimingMode::Throughput;  // Default: production-like
    HarnessDTypeConfig dtype = GetDTypeFromEnv();    // Default: fp32 or YALI_DTYPE env

    // Parse arguments
    if (argc > 1)
        elemCount = atol(argv[1]);
    if (argc > 2)
        numCalls = atoi(argv[2]);
    if (argc > 3)
        verify = (atoi(argv[3]) != 0);
    if (argc > 4) {
        if (strcmp(argv[4], "flash") == 0)
            mode = KernelMode::Flash;
        else if (strcmp(argv[4], "stream") == 0)
            mode = KernelMode::Stream;
    }
    if (argc > 5)
        lanesOverride = atoi(argv[5]);
    if (argc > 6) {
        if (strcmp(argv[6], "latency") == 0)
            timingMode = TimingMode::Latency;
        else if (strcmp(argv[6], "throughput") == 0)
            timingMode = TimingMode::Throughput;
        else if (strcmp(argv[6], "cuda-events") == 0 || strcmp(argv[6], "events") == 0)
            timingMode = TimingMode::CudaEvents;
    }
    // Optional 7th arg: dtype override (fp32, fp16, bf16)
    if (argc > 7) {
        dtype = ParseDType(argv[7]);
    }

    const size_t bytes = elemCount * dtype.elementSize;
    const size_t crossover = yali::FlashCrossoverBytes(dtype.tuningDtype);

    // Auto-select mode based on size
    bool useFlash;
    if (mode == KernelMode::Flash) {
        useFlash = true;
    } else if (mode == KernelMode::Stream) {
        useFlash = false;
    } else {
        useFlash = (bytes <= crossover);
    }

    // Print usage if requested
    if (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        printf("Usage: %s [elements] [calls] [verify] [mode] [lanes] [timing] [dtype]\n", argv[0]);
        printf("\n");
        printf("Arguments:\n");
        printf("  elements   Number of elements (default: 262144 = 1MB for fp32)\n");
        printf("  calls      Number of AllReduce calls to benchmark (default: 1000)\n");
        printf("  verify     Enable correctness check: 0 or 1 (default: 0)\n");
        printf("  mode       Kernel mode: auto, flash, stream (default: auto)\n");
        printf("  lanes      Lane count override: 0=auto (default: 0)\n");
        printf("  timing     Timing mode: throughput, latency, cuda-events (default: throughput)\n");
        printf("  dtype      Data type: fp32, fp16, bf16 (default: fp32 or YALI_DTYPE env)\n");
        printf("\n");
        printf("Timing modes:\n");
        printf("  throughput   Wall-clock, fire-and-forget, single sync at end (inference-like)\n");
        printf("  latency      Wall-clock, sync after each call (includes driver overhead, BS=1)\n");
        printf("  cuda-events  CUDA events around batch (GPU-only, matches ThunderKittens)\n");
        printf("\n");
        printf("Environment variables:\n");
        printf("  YALI_DTYPE   Override data type (fp32, fp16, bf16)\n");
        printf("\n");
        printf("Examples:\n");
        printf("  # 64MB Flash mode, fp32, throughput timing\n");
        printf("  %s 16777216 20 0 flash 0 throughput fp32\n", argv[0]);
        printf("\n");
        printf("  # 128MB Stream mode, fp16, CUDA events timing\n");
        printf("  %s 67108864 20 0 stream 0 cuda-events fp16\n", argv[0]);
        printf("\n");
        printf("  # 128MB Stream mode, bf16, verify enabled\n");
        printf("  %s 67108864 20 1 stream 0 throughput bf16\n", argv[0]);
        printf("\n");
        printf("  # Profile with nsys:\n");
        printf("  nsys profile -o stream_profile %s 33554432 20 0 stream\n", argv[0]);
        return 0;
    }

    printf("================================================================================\n");
    printf("YALI AllReduce Benchmark (%s)\n", dtype.name);
    printf("================================================================================\n");
    printf("Data type:    %s (element size: %zu bytes)\n", dtype.name, dtype.elementSize);
    printf("Elements:     %zu (%.2f MB)\n", elemCount, bytes / 1e6);
    printf("Calls:        %d (warmup: %d)\n", numCalls, warmupCalls);
    printf("Crossover:    %.0f MB (auto selects: %s)\n", crossover / 1e6, useFlash ? "flash" : "stream");
    printf("Kernel mode:  %s\n", useFlash ? "FLASH" : "STREAM");
    printf("Timing mode:  %s\n", TimingModeName(timingMode));
    if (lanesOverride > 0)
        printf("Lanes:        %d (override)\n", lanesOverride);
    if (verify)
        printf("Verification: ENABLED\n");
    printf("================================================================================\n\n");

    if (useFlash) {
        benchmarkFlash(elemCount, numCalls, warmupCalls, verify, dtype, lanesOverride, timingMode);
    } else {
        benchmarkStream(elemCount, numCalls, warmupCalls, verify, dtype, lanesOverride, timingMode);
    }

    return 0;
}
