/**
 * YALI AllReduce Benchmark (MPI)
 *
 * Multi-process version of benchmark_yali.cu.
 * Each MPI rank manages one GPU.
 *
 * Usage:
 *   mpirun -np 2 ./benchmark_yali_mpi <elements> <calls> [verify] [mode] [lanes] [timing]
 *   timing: throughput (default), latency, cuda-events
 *
 * This benchmark mimics how inference engines actually use AllReduce:
 * 1. Setup communicators/buffers once
 * 2. Run N allreduce calls in a tight loop (no sync between)
 * 3. Sync only at end
 * 4. Measure total throughput
 *
 * Supports both Flash mode (small messages) and Stream mode (large messages).
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// Comm infrastructure
#include "src/comm/comm.h"
#include "src/comm/ipc.cuh"

// Public headers from src/include/
#include "yali_launch.h"
#include "yali_tuning.h"

// AllReduce kernels
#include "src/all_reduce/kernels.cuh"

// Common utilities
#include "src/common/buffer_ops.cuh"
#include "src/common/hw_info.cuh"
#include "src/common/validation.cuh"

// Stream kernel entry point
extern "C" __global__ void _YaliKernel(YaliLaunchArgs args);

#define CHECK_CUDA(cmd)                                                                                                \
    do {                                                                                                               \
        cudaError_t e = cmd;                                                                                           \
        if (e != cudaSuccess) {                                                                                        \
            fprintf(stderr, "[Rank %d] CUDA error %s:%d: %s\n", myRank_, __FILE__, __LINE__, cudaGetErrorString(e));   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                              \
        }                                                                                                              \
    } while (0)

#define CHECK_MPI(cmd)                                                                                                 \
    do {                                                                                                               \
        int r = cmd;                                                                                                   \
        if (r != MPI_SUCCESS) {                                                                                        \
            fprintf(stderr, "MPI error %s:%d: %d\n", __FILE__, __LINE__, r);                                           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                              \
        }                                                                                                              \
    } while (0)

// Global rank for error messages (set in main)
static int myRank_ = 0;

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
// Data type configuration (matches benchmark_yali.cu)
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
    yali::DType tuningDtype;
};

static HarnessDTypeConfig ParseDType(const char* dtypeStr) {
    std::string lowered = dtypeStr ? std::string(dtypeStr) : std::string("fp32");
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

//------------------------------------------------------------------------------
// Flash Mode Benchmark (MPI) - Templated for multi-dtype support
//------------------------------------------------------------------------------
template <typename T>
void benchmarkFlashTyped(YaliMPComm* comm, size_t elemCount, int numCalls, int warmupCalls, bool verify,
                         const HarnessDTypeConfig& dtype, int lanesOverride, TimingMode timingMode) {
    const int myRank = comm->rank;
    const int worldSize = comm->worldSize;
    const int peerRank = 1 - myRank;  // For 2-rank
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

    // Allocate local buffers
    T* send = nullptr;
    T* recv = nullptr;
    CHECK_CUDA(cudaMalloc(&send, bytes));
    CHECK_CUDA(cudaMalloc(&recv, bytes));
    CHECK_CUDA(yali::SeedBufferSync(send, elemCount, static_cast<T>(myRank + 1)));
    CHECK_CUDA(cudaMemset(recv, 0, bytes));

    // Exchange IPC handles to get peer's send buffer
    if (yaliIpcExchangeBuffers(comm, send, bytes) != 0) {
        fprintf(stderr, "[Rank %d] Failed to exchange IPC handles\n", myRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    void* peerSend = comm->peerPtrs[peerRank];

    // Create stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Set shared memory attribute
    CHECK_CUDA(cudaFuncSetAttribute((const void*)yali::FlashKernel<T, 3>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sharedBytes)));

    // Setup launch args for each lane
    std::vector<YaliLaunchArgs> hostArgs(lanes);
    YaliLaunchArgs* deviceArgs = nullptr;
    CHECK_CUDA(cudaMalloc(&deviceArgs, lanes * sizeof(YaliLaunchArgs)));

    for (int lane = 0; lane < lanes; lane++) {
        size_t startElem = static_cast<size_t>(lane) * baseLaneElems;
        size_t endElem = std::min(startElem + baseLaneElems, elemCount);
        size_t laneElems = (endElem > startElem) ? (endElem - startElem) : 0;
        size_t offsetBytes = startElem * dtype.elementSize;

        auto& args = hostArgs[lane];
        args = {};
        args.localInput = send;
        args.localOutput = recv;
        args.peerInput = peerSend;
        args.elementCount = laneElems;
        args.elementSize = dtype.elementSize;
        args.sendOffset = offsetBytes;
        args.recvOffset = offsetBytes;
        args.datatype = dtype.ncclType;
        args.redOp = ncclSum;
        args.rank = myRank;
        args.worldSize = worldSize;
        args.laneIndex = lane;
        args.laneCount = lanes;
        args.ctasPerLane = ctasPerLane;
        args.flash = 1;
    }
    CHECK_CUDA(cudaMemcpy(deviceArgs, hostArgs.data(), lanes * sizeof(YaliLaunchArgs), cudaMemcpyHostToDevice));

    const dim3 grid(lanes * ctasPerLane);
    const dim3 block(blockSize);

    if (myRank == 0) {
        printf("Mode: FLASH (MPI, %s) | lanes=%d, ctasPerLane=%d, grid=%d, block=%d\n", dtype.name, lanes, ctasPerLane,
               grid.x, block.x);
        printf("Timing mode: %s\n", TimingModeName(timingMode));
    }

    // Lambda for launching one iteration
    auto launchIteration = [&]() {
        yali::FlashKernel<T, 3><<<grid, block, sharedBytes, stream>>>(deviceArgs, lanes, ctasPerLane);
    };

    // Sync stream
    auto syncStream = [&]() { CHECK_CUDA(cudaStreamSynchronize(stream)); };

    // Warmup
    for (int iter = 0; iter < warmupCalls; iter++) {
        launchIteration();
    }
    syncStream();
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

    // Timed run
    double localMs = 0.0;

    if (timingMode == TimingMode::CudaEvents) {
        cudaEvent_t startEvent, stopEvent;
        CHECK_CUDA(cudaEventCreate(&startEvent));
        CHECK_CUDA(cudaEventCreate(&stopEvent));

        syncStream();
        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

        CHECK_CUDA(cudaEventRecord(startEvent, stream));

        for (int iter = 0; iter < numCalls; iter++) {
            launchIteration();
        }

        CHECK_CUDA(cudaEventRecord(stopEvent, stream));
        syncStream();

        float elapsedMs = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
        localMs = static_cast<double>(elapsedMs);

        CHECK_CUDA(cudaEventDestroy(startEvent));
        CHECK_CUDA(cudaEventDestroy(stopEvent));

    } else if (timingMode == TimingMode::Throughput) {
        syncStream();
        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

        auto start = std::chrono::steady_clock::now();

        for (int iter = 0; iter < numCalls; iter++) {
            launchIteration();
        }
        syncStream();

        auto end = std::chrono::steady_clock::now();
        localMs = std::chrono::duration<double, std::milli>(end - start).count();

    } else {
        syncStream();
        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

        auto start = std::chrono::steady_clock::now();

        for (int iter = 0; iter < numCalls; iter++) {
            launchIteration();
            syncStream();
        }

        auto end = std::chrono::steady_clock::now();
        localMs = std::chrono::duration<double, std::milli>(end - start).count();
    }

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

    // Get global max time
    double globalMaxMs = 0.0;
    CHECK_MPI(MPI_Allreduce(&localMs, &globalMaxMs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

    double avgUs = (globalMaxMs * 1000.0) / numCalls;
    double dataBytes = static_cast<double>(bytes);
    double busBwFactor = 2.0 * static_cast<double>(worldSize - 1) / static_cast<double>(worldSize);
    double gbps = (dataBytes * busBwFactor / 1e9) / (avgUs / 1e6);
    double solPercent = gbps / 100.0 * 100.0;

    if (myRank == 0) {
        const char* modeStr = (timingMode == TimingMode::CudaEvents)   ? "cuda-events"
                              : (timingMode == TimingMode::Throughput) ? "throughput"
                                                                       : "latency";
        printf("YALI MPI (Flash-%s, %s): %d calls, %.2f us/call avg, %.2f GB/s (%.1f%% SoL)\n", dtype.name, modeStr,
               numCalls, avgUs, gbps, solPercent);
    }

    // Verification
    if (verify) {
        bool rankOk = yali::ValidateRankResult(recv, elemCount, myRank, worldSize);
        int localOk = rankOk ? 1 : 0;
        int globalOk = 0;
        CHECK_MPI(MPI_Allreduce(&localOk, &globalOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        if (myRank == 0) {
            printf("Correctness: %s\n", (globalOk == 1) ? "PASSED" : "FAILED");
        }
    }

    // Cleanup
    cudaFree(deviceArgs);
    cudaFree(send);
    cudaFree(recv);
    cudaStreamDestroy(stream);
}

// Dispatch function for Flash benchmark (selects template based on dtype)
void benchmarkFlash(YaliMPComm* comm, size_t elemCount, int numCalls, int warmupCalls, bool verify,
                    const HarnessDTypeConfig& dtype, int lanesOverride, TimingMode timingMode) {
    switch (dtype.kind) {
        case HarnessDTypeKind::kFloat16:
            benchmarkFlashTyped<__half>(comm, elemCount, numCalls, warmupCalls, verify, dtype, lanesOverride,
                                        timingMode);
            break;
        case HarnessDTypeKind::kBFloat16:
            benchmarkFlashTyped<__nv_bfloat16>(comm, elemCount, numCalls, warmupCalls, verify, dtype, lanesOverride,
                                               timingMode);
            break;
        case HarnessDTypeKind::kFloat32:
        default:
            benchmarkFlashTyped<float>(comm, elemCount, numCalls, warmupCalls, verify, dtype, lanesOverride,
                                       timingMode);
            break;
    }
}

//------------------------------------------------------------------------------
// Stream Mode Benchmark (MPI) - Templated for multi-dtype support
//------------------------------------------------------------------------------
template <typename T>
void benchmarkStreamTyped(YaliMPComm* comm, size_t elemCount, int numCalls, int warmupCalls, bool verify,
                          const HarnessDTypeConfig& dtype, int lanesOverride, TimingMode timingMode) {
    const int myRank = comm->rank;
    const int worldSize = comm->worldSize;
    const int peerRank = 1 - myRank;
    const size_t bytes = elemCount * dtype.elementSize;

    // Stream kernel config
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

    // Allocate send/recv buffers
    T* send = nullptr;
    T* recv = nullptr;
    CHECK_CUDA(cudaMalloc(&send, bytes));
    CHECK_CUDA(cudaMalloc(&recv, bytes));
    CHECK_CUDA(yali::SeedBufferSync(send, elemCount, static_cast<T>(myRank + 1)));
    CHECK_CUDA(cudaMemset(recv, 0, bytes));

    // Exchange IPC handles for send buffers (for peerInput)
    if (yaliIpcExchangeBuffers(comm, send, bytes) != 0) {
        fprintf(stderr, "[Rank %d] Failed to exchange IPC handles\n", myRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    void* peerSend = comm->peerPtrs[peerRank];

    // Create per-lane streams
    std::vector<cudaStream_t> laneStreams(lanes);
    for (int lane = 0; lane < lanes; lane++) {
        CHECK_CUDA(cudaStreamCreate(&laneStreams[lane]));
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

    // Allocate ring buffers for each lane
    std::vector<ManagedRing> laneRing(lanes);
    for (int lane = 0; lane < lanes; lane++) {
        size_t laneElems = laneElements[lane];
        size_t laneBytes = laneElems * dtype.elementSize;

        size_t laneCapacity = (laneBytes + ringSlotBytes - 1) / ringSlotBytes;
        if (laneCapacity == 0)
            laneCapacity = 1;
        if (laneBytes > 0 && laneCapacity < 4)
            laneCapacity = 4;

        laneRing[lane].capacity = static_cast<int>(laneCapacity);
        laneRing[lane].sequenceBytes = laneCapacity * sizeof(uint64_t);
        laneRing[lane].dataBytes = laneCapacity * ringSlotBytes;

        if (laneElems == 0) {
            laneRing[lane].sequence = nullptr;
            laneRing[lane].gating = nullptr;
            laneRing[lane].data = nullptr;
            continue;
        }

        CHECK_CUDA(cudaMalloc(&laneRing[lane].sequence, laneRing[lane].sequenceBytes));
        CHECK_CUDA(cudaMalloc(&laneRing[lane].gating, sizeof(uint64_t)));
        CHECK_CUDA(cudaMalloc(&laneRing[lane].data, laneRing[lane].dataBytes));
        CHECK_CUDA(cudaMemset(laneRing[lane].sequence, 0xff, laneRing[lane].sequenceBytes));
        CHECK_CUDA(cudaMemset(laneRing[lane].gating, 0, sizeof(uint64_t)));
    }

    // Exchange ring buffer IPC handles
    void** peerSequence = nullptr;
    void** peerGating = nullptr;
    void** peerData = nullptr;
    if (yaliIpcExchangeRingBuffers(comm, laneRing.data(), lanes, &peerSequence, &peerGating, &peerData) != 0) {
        fprintf(stderr, "[Rank %d] Failed to exchange ring IPC handles\n", myRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Setup launch args for each lane
    std::vector<YaliLaunchArgs> launchArgs(lanes);
    for (int lane = 0; lane < lanes; lane++) {
        size_t elems = laneElements[lane];
        size_t offsetElems = laneOffsets[lane];
        int peerIdx = peerRank * lanes + lane;

        auto& args = launchArgs[lane];
        args = {};

        // Send to peer's ring buffers
        args.sendSequence = reinterpret_cast<uint64_t*>(peerSequence[peerIdx]);
        args.sendGating = reinterpret_cast<uint64_t*>(peerGating[peerIdx]);
        args.sendData = reinterpret_cast<char*>(peerData[peerIdx]);
        args.sendCapacity = laneRing[lane].capacity;
        args.sendSlotBytes = ringSlotBytesInt;
        args.sendSlotStride = ringSlotBytesInt;

        // Receive from own ring buffers
        args.recvSequence = laneRing[lane].sequence;
        args.recvGating = laneRing[lane].gating;
        args.recvData = laneRing[lane].data;
        args.recvCapacity = laneRing[lane].capacity;
        args.recvSlotBytes = ringSlotBytesInt;
        args.recvSlotStride = ringSlotBytesInt;

        args.localInput = reinterpret_cast<char*>(send) + offsetElems * dtype.elementSize;
        args.localOutput = reinterpret_cast<char*>(recv) + offsetElems * dtype.elementSize;
        args.peerInput = reinterpret_cast<char*>(peerSend) + offsetElems * dtype.elementSize;
        args.elementCount = elems;
        args.elementSize = dtype.elementSize;
        args.sendOffset = 0;
        args.recvOffset = 0;
        args.initialSequence = 0;
        args.datatype = dtype.ncclType;
        args.redOp = ncclSum;
        args.rank = myRank;
        args.worldSize = worldSize;
        args.laneIndex = lane;
        args.laneCount = lanes;
        args.ctasPerLane = ctasPerLane;
        args.flash = 0;
    }

    const dim3 grid(1);
    const dim3 block(blockSize);

    if (myRank == 0) {
        printf("Mode: STREAM (MPI, %s) | lanes=%d, blockSize=%d\n", dtype.name, lanes, blockSize);
        printf("Timing mode: %s\n", TimingModeName(timingMode));
    }

    // Track sequence base across iterations
    uint64_t globalIterCount = 0;

    // Lambda to launch one iteration
    auto launchIteration = [&](uint64_t iterIdx) {
        // Pre-compute sequence base
        for (int lane = 0; lane < lanes; lane++) {
            launchArgs[lane].initialSequence = iterIdx * laneSlotsUsed[lane];
        }

        // Launch kernels for all lanes
        for (int lane = 0; lane < lanes; lane++) {
            if (laneElements[lane] == 0)
                continue;
            void* kernelParams[] = {&launchArgs[lane]};
            CHECK_CUDA(cudaLaunchKernel((const void*)_YaliKernel, grid, block, kernelParams, 0, laneStreams[lane]));
        }
    };

    // Sync all lanes
    auto syncAll = [&]() {
        for (int lane = 0; lane < lanes; lane++) {
            if (laneElements[lane] == 0)
                continue;
            CHECK_CUDA(cudaStreamSynchronize(laneStreams[lane]));
        }
    };

    // Warmup
    for (int iter = 0; iter < warmupCalls; iter++) {
        launchIteration(globalIterCount++);
        syncAll();
    }
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

    // Timed run
    double localMs = 0.0;

    if (timingMode == TimingMode::CudaEvents) {
        cudaEvent_t startEvent, stopEvent;
        CHECK_CUDA(cudaEventCreate(&startEvent));
        CHECK_CUDA(cudaEventCreate(&stopEvent));

        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

        // Record on first lane's stream
        CHECK_CUDA(cudaEventRecord(startEvent, laneStreams[0]));

        for (int iter = 0; iter < numCalls; iter++) {
            launchIteration(globalIterCount++);
        }

        CHECK_CUDA(cudaEventRecord(stopEvent, laneStreams[0]));
        syncAll();

        float elapsedMs = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
        localMs = static_cast<double>(elapsedMs);

        CHECK_CUDA(cudaEventDestroy(startEvent));
        CHECK_CUDA(cudaEventDestroy(stopEvent));

    } else if (timingMode == TimingMode::Throughput) {
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

        auto start = std::chrono::steady_clock::now();

        for (int iter = 0; iter < numCalls; iter++) {
            launchIteration(globalIterCount++);
        }
        syncAll();

        auto end = std::chrono::steady_clock::now();
        localMs = std::chrono::duration<double, std::milli>(end - start).count();

    } else {
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

        auto start = std::chrono::steady_clock::now();

        for (int iter = 0; iter < numCalls; iter++) {
            launchIteration(globalIterCount++);
            syncAll();
        }

        auto end = std::chrono::steady_clock::now();
        localMs = std::chrono::duration<double, std::milli>(end - start).count();
    }

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

    // Get global max time
    double globalMaxMs = 0.0;
    CHECK_MPI(MPI_Allreduce(&localMs, &globalMaxMs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

    double avgUs = (globalMaxMs * 1000.0) / numCalls;
    double dataBytes = static_cast<double>(bytes);
    double busBwFactor = 2.0 * static_cast<double>(worldSize - 1) / static_cast<double>(worldSize);
    double gbps = (dataBytes * busBwFactor / 1e9) / (avgUs / 1e6);
    double solPercent = gbps / 100.0 * 100.0;

    if (myRank == 0) {
        const char* modeStr = (timingMode == TimingMode::CudaEvents)   ? "cuda-events"
                              : (timingMode == TimingMode::Throughput) ? "throughput"
                                                                       : "latency";
        printf("YALI MPI (Stream-%s, %s): %d calls, %.2f us/call avg, %.2f GB/s (%.1f%% SoL)\n", dtype.name, modeStr,
               numCalls, avgUs, gbps, solPercent);
    }

    // Verification
    if (verify) {
        bool rankOk = yali::ValidateRankResult(recv, elemCount, myRank, worldSize);
        int localOk = rankOk ? 1 : 0;
        int globalOk = 0;
        CHECK_MPI(MPI_Allreduce(&localOk, &globalOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        if (myRank == 0) {
            printf("Correctness: %s\n", (globalOk == 1) ? "PASSED" : "FAILED");
        }
    }

    // Cleanup
    if (peerSequence || peerGating || peerData) {
        yaliIpcCloseRingBuffers(comm, lanes, peerSequence, peerGating, peerData);
    }
    for (int lane = 0; lane < lanes; lane++) {
        if (laneRing[lane].sequence)
            cudaFree(laneRing[lane].sequence);
        if (laneRing[lane].gating)
            cudaFree(laneRing[lane].gating);
        if (laneRing[lane].data)
            cudaFree(laneRing[lane].data);
        cudaStreamDestroy(laneStreams[lane]);
    }
    cudaFree(send);
    cudaFree(recv);
}

// Dispatch function for Stream benchmark (selects template based on dtype)
void benchmarkStream(YaliMPComm* comm, size_t elemCount, int numCalls, int warmupCalls, bool verify,
                     const HarnessDTypeConfig& dtype, int lanesOverride, TimingMode timingMode) {
    switch (dtype.kind) {
        case HarnessDTypeKind::kFloat16:
            benchmarkStreamTyped<__half>(comm, elemCount, numCalls, warmupCalls, verify, dtype, lanesOverride,
                                         timingMode);
            break;
        case HarnessDTypeKind::kBFloat16:
            benchmarkStreamTyped<__nv_bfloat16>(comm, elemCount, numCalls, warmupCalls, verify, dtype, lanesOverride,
                                                timingMode);
            break;
        case HarnessDTypeKind::kFloat32:
        default:
            benchmarkStreamTyped<float>(comm, elemCount, numCalls, warmupCalls, verify, dtype, lanesOverride,
                                        timingMode);
            break;
    }
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Initialize MPI communicator
    YaliMPComm* comm = yaliMPCommCreate(&argc, &argv);
    if (!comm) {
        fprintf(stderr, "Failed to create MPI communicator\n");
        return EXIT_FAILURE;
    }

    myRank_ = comm->rank;  // For error messages

    // Validate world size
    if (comm->worldSize != 2) {
        if (comm->rank == 0) {
            fprintf(stderr, "Error: This benchmark requires exactly 2 MPI ranks (got %d)\n", comm->worldSize);
        }
        yaliMPCommDestroy(comm);
        return EXIT_FAILURE;
    }

    // Parse arguments
    size_t elemCount = 262144;  // 1MB (default)
    int numCalls = 1000;
    int warmupCalls = 100;
    bool verify = false;
    KernelMode mode = KernelMode::Auto;
    int lanesOverride = 0;
    TimingMode timingMode = TimingMode::Throughput;
    HarnessDTypeConfig dtype = GetDTypeFromEnv();  // Default: fp32 or YALI_DTYPE env

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

    // Auto-select mode
    bool useFlash;
    if (mode == KernelMode::Flash) {
        useFlash = true;
    } else if (mode == KernelMode::Stream) {
        useFlash = false;
    } else {
        useFlash = (bytes <= crossover);
    }

    // Print header
    if (comm->rank == 0) {
        printf("================================================================================\n");
        printf("YALI AllReduce Benchmark (MPI, %d ranks, %s)\n", comm->worldSize, dtype.name);
        printf("================================================================================\n");
        printf("Data type:    %s (element size: %zu bytes)\n", dtype.name, dtype.elementSize);
        printf("Elements:     %zu (%.2f MB)\n", elemCount, static_cast<double>(bytes) / 1e6);
        printf("Calls:        %d (warmup: %d)\n", numCalls, warmupCalls);
        printf("Crossover:    %.0f MB (auto selects: %s)\n", crossover / 1e6, useFlash ? "flash" : "stream");
        printf("Kernel mode:  %s\n", useFlash ? "FLASH" : "STREAM");
        printf("Timing mode:  %s\n", TimingModeName(timingMode));
        if (lanesOverride > 0)
            printf("Lanes:        %d (override)\n", lanesOverride);
        if (verify)
            printf("Verification: ENABLED\n");
        printf("================================================================================\n\n");
    }

    if (useFlash) {
        benchmarkFlash(comm, elemCount, numCalls, warmupCalls, verify, dtype, lanesOverride, timingMode);
    } else {
        benchmarkStream(comm, elemCount, numCalls, warmupCalls, verify, dtype, lanesOverride, timingMode);
    }

    if (comm->rank == 0) {
        printf("\nUsage: mpirun -np 2 %s <elements> <calls> [verify] [mode] [lanes] [timing] [dtype]\n", argv[0]);
        printf("  mode: auto, flash, stream\n");
        printf("  timing: throughput (default), latency, cuda-events\n");
        printf("  dtype: fp32 (default), fp16, bf16\n");
    }

    yaliMPCommDestroy(comm);
    return EXIT_SUCCESS;
}
