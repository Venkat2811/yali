/*************************************************************************
 * Multi-Lane MPI AllReduce Example
 *
 * Full-featured 2-process AllReduce with explicit control over lane
 * configuration and kernel parameters. Use this when you need:
 *   - Custom lane counts for specific message sizes
 *   - Multiple data type support (FP32, FP16, BF16)
 *   - Direct control over YaliLaunchArgs
 *
 * For a simpler API using yali::MPIComm, see simple_mpi.cu in this
 * directory which uses src/ops/allreduce_mpi.cuh.
 *
 * Build:   bazel build //:example_multilane_mpi
 * Run:     CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 --allow-run-as-root bazel-bin/example_multilane_mpi
 *
 * Environment variables:
 *   YALI_DTYPE  - Data type: fp32 (default), fp16, bf16
 *   YALI_ELEMS  - Element count (default: 1048576)
 *   YALI_LANES  - Lane count (default: auto-selected)
 *
 * Features:
 *   - Multi-lane parallelism with configurable lane count
 *   - FP32/FP16/BF16 datatype support
 *   - Low-level YaliLaunchArgs control
 *   - IPC handle exchange for cross-process memory access
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/all_reduce/kernels.cuh"
#include "src/comm/comm.h"
#include "src/comm/ipc.cuh"
#include "src/common/buffer_ops.cuh"
#include "src/common/validation.cuh"
#include "yali_launch.h"

#define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = (call);                                                                                      \
        if (err != cudaSuccess) {                                                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

enum class DType { FP32, FP16, BF16 };

struct Config {
    DType dtype = DType::FP32;
    size_t elemCount = 1024 * 1024;
    int lanes = 0;  // 0 = auto
    const char* dtypeName = "fp32";
    size_t elemSize = sizeof(float);
};

static Config parseConfig() {
    Config cfg;

    // Parse dtype
    const char* dtypeEnv = std::getenv("YALI_DTYPE");
    if (dtypeEnv) {
        std::string dt(dtypeEnv);
        std::transform(dt.begin(), dt.end(), dt.begin(), [](unsigned char c) { return std::tolower(c); });
        if (dt == "fp16" || dt == "f16" || dt == "half") {
            cfg.dtype = DType::FP16;
            cfg.dtypeName = "fp16";
            cfg.elemSize = sizeof(__half);
        } else if (dt == "bf16" || dt == "bfloat16") {
            cfg.dtype = DType::BF16;
            cfg.dtypeName = "bf16";
            cfg.elemSize = sizeof(__nv_bfloat16);
        }
    }

    // Parse element count
    const char* elemsEnv = std::getenv("YALI_ELEMS");
    if (elemsEnv) {
        cfg.elemCount = std::strtoull(elemsEnv, nullptr, 10);
    }

    // Parse lanes
    const char* lanesEnv = std::getenv("YALI_LANES");
    if (lanesEnv) {
        cfg.lanes = std::atoi(lanesEnv);
    }

    return cfg;
}

// -----------------------------------------------------------------------------
// Template kernel launcher
// -----------------------------------------------------------------------------

template <typename T>
static bool runAllReduce(YaliMPComm* comm, size_t elemCount, int lanes) {
    const int rank = comm->rank;
    const int worldSize = comm->worldSize;
    const size_t bytes = elemCount * sizeof(T);

    // Allocate buffers
    T *sendBuf = nullptr, *recvBuf = nullptr;
    CHECK_CUDA(cudaMalloc(&sendBuf, bytes));
    CHECK_CUDA(cudaMalloc(&recvBuf, bytes));

    // Seed: rank 0 = 1.0, rank 1 = 2.0
    float seedValue = static_cast<float>(rank + 1);
    CHECK_CUDA(yali::SeedBufferSync(sendBuf, elemCount, seedValue));
    CHECK_CUDA(cudaMemset(recvBuf, 0, bytes));

    // Exchange IPC handles
    // After this call, comm->peerPtrs[i] contains pointer to rank i's buffer
    if (yaliIpcExchangeBuffers(comm, sendBuf, bytes) != 0) {
        fprintf(stderr, "Rank %d: IPC exchange failed\n", rank);
        return false;
    }
    void* peerSend = comm->peerPtrs[(rank + 1) % worldSize];

    // Kernel config
    constexpr int kPrefetchStages = 3;
    const int kBlockSize = 512;
    const int kCtasPerLane = 4;

    // Auto-select lanes if not specified
    if (lanes <= 0) {
        lanes = (bytes > 1024 * 1024) ? 16 : 4;
    }

    // Setup launch args
    size_t baseLaneElems = (elemCount + lanes - 1) / lanes;
    std::vector<YaliLaunchArgs> hostArgs(lanes);

    for (int lane = 0; lane < lanes; ++lane) {
        size_t startElem = static_cast<size_t>(lane) * baseLaneElems;
        size_t endElem = (startElem + baseLaneElems > elemCount) ? elemCount : startElem + baseLaneElems;
        size_t elems = (endElem > startElem) ? (endElem - startElem) : 0;
        size_t offset = startElem * sizeof(T);

        auto& args = hostArgs[lane];
        args = {};
        args.localInput = sendBuf;
        args.localOutput = recvBuf;
        args.peerInput = peerSend;
        args.elementCount = elems;
        args.elementSize = sizeof(T);
        args.sendOffset = offset;
        args.recvOffset = offset;
        args.rank = rank;
        args.laneIndex = lane;
        args.laneCount = lanes;
        args.ctasPerLane = kCtasPerLane;
        args.flash = 1;
    }

    // Copy to device
    YaliLaunchArgs* deviceArgs = nullptr;
    CHECK_CUDA(cudaMalloc(&deviceArgs, lanes * sizeof(YaliLaunchArgs)));
    CHECK_CUDA(cudaMemcpy(deviceArgs, hostArgs.data(), lanes * sizeof(YaliLaunchArgs), cudaMemcpyHostToDevice));

    // Configure shared memory
    size_t smemBytes = yali::FlashConfig::SharedMemoryBytes(kBlockSize, kPrefetchStages, sizeof(T));
    CHECK_CUDA(cudaFuncSetAttribute((const void*)yali::FlashKernel<T, kPrefetchStages>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smemBytes)));

    // Sync and launch
    yaliMPCommBarrier(comm);

    dim3 grid(lanes * kCtasPerLane);
    dim3 block(kBlockSize);

    yali::FlashKernel<T, kPrefetchStages><<<grid, block, smemBytes>>>(deviceArgs, lanes, kCtasPerLane);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    yaliMPCommBarrier(comm);

    // Validate
    bool valid = yali::ValidateRankResult(recvBuf, elemCount, rank, worldSize);
    printf("Rank %d: %s\n", rank, valid ? "OK" : "FAILED");

    // Cleanup
    // IPC handles are automatically closed by yaliMPCommDestroy
    cudaFree(sendBuf);
    cudaFree(recvBuf);
    cudaFree(deviceArgs);

    return valid;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    // Initialize MPI
    YaliMPComm* comm = yaliMPCommCreate(&argc, &argv);
    if (!comm) {
        fprintf(stderr, "Failed to create MPI communicator\n");
        return EXIT_FAILURE;
    }

    if (comm->worldSize != 2) {
        if (comm->rank == 0) {
            fprintf(stderr, "Error: This example requires exactly 2 MPI ranks (got %d)\n", comm->worldSize);
        }
        yaliMPCommDestroy(comm);
        return EXIT_FAILURE;
    }

    Config cfg = parseConfig();
    const int rank = comm->rank;

    if (rank == 0) {
        printf("=== Yali MPI Multi-Lane AllReduce Example ===\n");
        printf("World size: %d\n", comm->worldSize);
        printf("Data type:  %s\n", cfg.dtypeName);
        printf("Elements:   %zu (%.2f MB)\n", cfg.elemCount, cfg.elemCount * cfg.elemSize / 1e6);
        printf("Lanes:      %s\n", cfg.lanes > 0 ? std::to_string(cfg.lanes).c_str() : "auto");
    }

    bool success = false;
    switch (cfg.dtype) {
        case DType::FP32:
            success = runAllReduce<float>(comm, cfg.elemCount, cfg.lanes);
            break;
        case DType::FP16:
            success = runAllReduce<__half>(comm, cfg.elemCount, cfg.lanes);
            break;
        case DType::BF16:
            success = runAllReduce<__nv_bfloat16>(comm, cfg.elemCount, cfg.lanes);
            break;
    }

    if (rank == 0) {
        printf("=== Example %s ===\n", success ? "PASSED" : "FAILED");
    }

    yaliMPCommDestroy(comm);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
