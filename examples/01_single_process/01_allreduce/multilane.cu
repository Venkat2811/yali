/*************************************************************************
 * Multi-Lane AllReduce Example
 *
 * Full-featured 2-GPU AllReduce with explicit control over lane
 * configuration and kernel parameters. Use this when you need:
 *   - Custom lane counts for specific message sizes
 *   - Direct control over YaliLaunchArgs
 *   - Integration with existing kernel launch infrastructure
 *
 * For a simpler API, see simple.cu in this directory.
 *
 * Build:   bazel build //examples/01_single_process/01_allreduce:multilane
 * Run:     CUDA_VISIBLE_DEVICES=0,1 bazel-bin/examples/01_single_process/01_allreduce/multilane
 *
 * Features:
 *   - Multi-lane parallelism with configurable lane count
 *   - Manual YaliLaunchArgs setup per GPU
 *   - Direct kernel template instantiation
 ************************************************************************/

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "src/all_reduce/kernels.cuh"
#include "src/common/buffer_ops.cuh"
#include "src/common/peer_access.cuh"
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

int main() {
    // Configuration
    constexpr int kRanks = 2;
    constexpr size_t kElemCount = 1024 * 1024;  // 1M elements = 4MB per GPU
    constexpr size_t kBytes = kElemCount * sizeof(float);
    constexpr int kLanes = 16;
    constexpr int kCtasPerLane = 1;
    constexpr int kBlockSize = 512;
    constexpr int kPrefetchStages = 3;

    // Check we have 2 GPUs
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        printf("Need 2 GPUs, found %d\n", deviceCount);
        return EXIT_SUCCESS;
    }

    // Enable P2P access
    yali::EnablePeerAccessOrDie(0, 1);
    yali::EnablePeerAccessOrDie(1, 0);
    printf("P2P enabled between GPU 0 and GPU 1\n");

    // Allocate send/recv buffers on each GPU
    std::vector<float*> send(kRanks), recv(kRanks);
    for (int r = 0; r < kRanks; ++r) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaMalloc(&send[r], kBytes));
        CHECK_CUDA(cudaMalloc(&recv[r], kBytes));

        // Seed: rank 0 gets all 1.0, rank 1 gets all 2.0
        float seedValue = static_cast<float>(r + 1);
        CHECK_CUDA(yali::SeedBufferSync(send[r], kElemCount, seedValue));
        CHECK_CUDA(cudaMemset(recv[r], 0, kBytes));
    }
    printf("Allocated and seeded %zu elements per GPU\n", kElemCount);

    // Setup launch args for each rank
    // For low-latency kernel, we only need a subset of YaliLaunchArgs
    std::vector<std::vector<YaliLaunchArgs>> launchArgs(kRanks);
    std::vector<YaliLaunchArgs*> deviceArgs(kRanks, nullptr);

    size_t baseLaneElems = (kElemCount + kLanes - 1) / kLanes;
    for (int r = 0; r < kRanks; ++r) {
        CHECK_CUDA(cudaSetDevice(r));
        launchArgs[r].resize(kLanes);

        for (int lane = 0; lane < kLanes; ++lane) {
            size_t startElem = static_cast<size_t>(lane) * baseLaneElems;
            size_t endElem = std::min(startElem + baseLaneElems, kElemCount);
            size_t elems = (endElem > startElem) ? (endElem - startElem) : 0;
            size_t offset = startElem * sizeof(float);

            auto& args = launchArgs[r][lane];
            args = {};
            args.localInput = send[r];
            args.localOutput = recv[r];
            args.peerInput = send[(r + 1) % kRanks];
            args.elementCount = elems;
            args.elementSize = sizeof(float);
            args.sendOffset = offset;
            args.recvOffset = offset;
            args.rank = r;
            args.laneIndex = lane;
            args.laneCount = kLanes;
            args.ctasPerLane = kCtasPerLane;
            args.flash = 1;
        }

        size_t argsBytes = kLanes * sizeof(YaliLaunchArgs);
        CHECK_CUDA(cudaMalloc(&deviceArgs[r], argsBytes));
        CHECK_CUDA(cudaMemcpy(deviceArgs[r], launchArgs[r].data(), argsBytes, cudaMemcpyHostToDevice));
    }

    // Calculate shared memory requirement
    size_t sharedBytes = yali::FlashConfig::SharedMemoryBytes(kBlockSize, kPrefetchStages, sizeof(float));

    // Set shared memory attribute on kernel
    for (int r = 0; r < kRanks; ++r) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaFuncSetAttribute((const void*)yali::FlashKernel<float, kPrefetchStages>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sharedBytes)));
    }

    // Launch kernel on each GPU
    const unsigned int totalBlocks = kLanes * kCtasPerLane;
    const dim3 grid(totalBlocks);
    const dim3 block(kBlockSize);

    printf("Launching AllReduce: %d lanes, %d CTAs/lane, %d threads/block\n", kLanes, kCtasPerLane, kBlockSize);

    for (int r = 0; r < kRanks; ++r) {
        CHECK_CUDA(cudaSetDevice(r));
        yali::FlashKernel<float, kPrefetchStages><<<grid, block, sharedBytes>>>(deviceArgs[r], kLanes, kCtasPerLane);
        CHECK_CUDA(cudaGetLastError());
    }

    // Synchronize all
    for (int r = 0; r < kRanks; ++r) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Validate results
    // Expected: all elements should be 1.0 + 2.0 = 3.0
    bool allOk = true;
    for (int r = 0; r < kRanks; ++r) {
        CHECK_CUDA(cudaSetDevice(r));
        bool rankOk = yali::ValidateRankResult(recv[r], kElemCount, r, kRanks);
        if (!rankOk) {
            printf("Rank %d: FAILED\n", r);
            allOk = false;
        } else {
            printf("Rank %d: OK\n", r);
        }
    }

    // Cleanup
    for (int r = 0; r < kRanks; ++r) {
        CHECK_CUDA(cudaSetDevice(r));
        cudaFree(send[r]);
        cudaFree(recv[r]);
        cudaFree(deviceArgs[r]);
    }

    printf("AllReduce example %s\n", allOk ? "PASSED" : "FAILED");
    return allOk ? EXIT_SUCCESS : EXIT_FAILURE;
}
