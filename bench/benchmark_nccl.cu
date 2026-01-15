/**
 * AllReduce Benchmark (NCCL) - benchmark_nccl
 *
 * This benchmark mimics how inference engines actually use AllReduce:
 * 1. Setup communicators/buffers once
 * 2. Run N allreduce calls in a tight loop (no sync between)
 * 3. Sync only at end
 * 4. Measure total throughput
 *
 * Multi-dtype support: fp32, fp16, bf16 (set via YALI_DTYPE env or --dtype arg)
 *
 * Supports three timing modes for fair comparison:
 * - throughput: Wall-clock, fire-and-forget (default, production-like)
 * - latency:    Wall-clock, sync after each call (BS=1 interactive)
 * - cuda-events: GPU-only timing (ThunderKittens methodology)
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <nccl.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK_CUDA(cmd)                                                                                                \
    do {                                                                                                               \
        cudaError_t e = cmd;                                                                                           \
        if (e != cudaSuccess) {                                                                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                      \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

#define CHECK_NCCL(cmd)                                                                                                \
    do {                                                                                                               \
        ncclResult_t r = cmd;                                                                                          \
        if (r != ncclSuccess) {                                                                                        \
            fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r));                      \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

//------------------------------------------------------------------------------
// Data Type Configuration (multi-dtype support: fp32, fp16, bf16)
//------------------------------------------------------------------------------
enum class NCCLDTypeKind {
    kFloat32 = 0,
    kFloat16 = 1,
    kBFloat16 = 2,
};

struct NCCLDTypeConfig {
    NCCLDTypeKind kind;
    ncclDataType_t ncclType;
    size_t elementSize;
    const char* name;
};

static NCCLDTypeConfig ParseDType(const char* dtypeStr) {
    std::string lowered = dtypeStr ? std::string(dtypeStr) : std::string("f32");
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lowered == "f16" || lowered == "fp16" || lowered == "float16") {
        return {NCCLDTypeKind::kFloat16, ncclHalf, sizeof(__half), "fp16"};
    }
    if (lowered == "bf16" || lowered == "bfloat16") {
        return {NCCLDTypeKind::kBFloat16, ncclBfloat16, sizeof(__nv_bfloat16), "bf16"};
    }
    return {NCCLDTypeKind::kFloat32, ncclFloat, sizeof(float), "fp32"};
}

static NCCLDTypeConfig GetDTypeFromEnv() {
    const char* env = std::getenv("YALI_DTYPE");
    return ParseDType(env);
}

//------------------------------------------------------------------------------
// Timing Mode (ThunderKittens-compatible)
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

void benchmarkNCCL(size_t elemCount, int numCalls, int warmupCalls, TimingMode timingMode,
                   const NCCLDTypeConfig& dtype) {
    const int nGpus = 2;
    const size_t bytes = elemCount * dtype.elementSize;

    // Setup - done once
    ncclComm_t comms[nGpus];
    cudaStream_t streams[nGpus];
    void* sendbuffs[nGpus];
    void* recvbuffs[nGpus];

    ncclUniqueId id;
    ncclGetUniqueId(&id);

    for (int i = 0; i < nGpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc(&sendbuffs[i], bytes));
        CHECK_CUDA(cudaMalloc(&recvbuffs[i], bytes));
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    CHECK_NCCL(ncclGroupStart());
    for (int i = 0; i < nGpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_NCCL(ncclCommInitRank(&comms[i], nGpus, id, i));
    }
    CHECK_NCCL(ncclGroupEnd());

    printf("Timing mode: %s\n", TimingModeName(timingMode));

    // Lambda for launching one iteration
    auto launchIteration = [&]() {
        CHECK_NCCL(ncclGroupStart());
        for (int i = 0; i < nGpus; i++) {
            CHECK_NCCL(
                ncclAllReduce(sendbuffs[i], recvbuffs[i], elemCount, dtype.ncclType, ncclSum, comms[i], streams[i]));
        }
        CHECK_NCCL(ncclGroupEnd());
    };

    // Sync all helper
    auto syncAll = [&]() {
        for (int i = 0; i < nGpus; i++) {
            CHECK_CUDA(cudaSetDevice(i));
            CHECK_CUDA(cudaStreamSynchronize(streams[i]));
        }
    };

    // Warmup - like real inference warmup
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
    const int nranks = nGpus;
    double dataBytes = static_cast<double>(bytes);
    double busBwFactor = 2.0 * static_cast<double>(nranks - 1) / static_cast<double>(nranks);
    double gbps = (dataBytes * busBwFactor / 1e9) / (avgUs / 1e6);
    double solPercent = gbps / 100.0 * 100.0;  // vs 100 GB/s unidirectional NVLink

    const char* modeStr = (timingMode == TimingMode::CudaEvents)   ? "cuda-events"
                          : (timingMode == TimingMode::Throughput) ? "throughput"
                                                                   : "latency";
    printf("NCCL (%s, %s): %d calls, %.2f us/call avg, %.2f GB/s (%.1f%% SoL)\n", dtype.name, modeStr, numCalls, avgUs,
           gbps, solPercent);

    // Cleanup
    for (int i = 0; i < nGpus; i++) {
        ncclCommDestroy(comms[i]);
        cudaFree(sendbuffs[i]);
        cudaFree(recvbuffs[i]);
        cudaStreamDestroy(streams[i]);
    }
}

int main(int argc, char** argv) {
    size_t elemCount = 262144;  // 1MB for fp32
    int numCalls = 1000;        // Like 1000 layers
    int warmupCalls = 100;
    TimingMode timingMode = TimingMode::Throughput;
    NCCLDTypeConfig dtype = GetDTypeFromEnv();  // Default: fp32 or YALI_DTYPE env

    if (argc > 1)
        elemCount = atol(argv[1]);
    if (argc > 2)
        numCalls = atoi(argv[2]);
    if (argc > 3) {
        if (strcmp(argv[3], "latency") == 0)
            timingMode = TimingMode::Latency;
        else if (strcmp(argv[3], "throughput") == 0)
            timingMode = TimingMode::Throughput;
        else if (strcmp(argv[3], "cuda-events") == 0 || strcmp(argv[3], "events") == 0)
            timingMode = TimingMode::CudaEvents;
    }
    // Optional 4th arg: dtype override (fp32, fp16, bf16)
    if (argc > 4) {
        dtype = ParseDType(argv[4]);
    }

    const size_t bytes = elemCount * dtype.elementSize;

    // Print usage if requested
    if (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        printf("Usage: %s [elements] [calls] [timing] [dtype]\n", argv[0]);
        printf("\n");
        printf("Arguments:\n");
        printf("  elements   Number of elements (default: 262144 = 1MB for fp32)\n");
        printf("  calls      Number of AllReduce calls to benchmark (default: 1000)\n");
        printf("  timing     Timing mode: throughput, latency, cuda-events (default: throughput)\n");
        printf("  dtype      Data type: fp32, fp16, bf16 (default: fp32 or YALI_DTYPE env)\n");
        printf("\n");
        printf("Environment variables:\n");
        printf("  YALI_DTYPE   Override data type (fp32, fp16, bf16)\n");
        printf("\n");
        printf("Examples:\n");
        printf("  %s 16777216 20 throughput fp32   # 64MB fp32\n", argv[0]);
        printf("  %s 67108864 20 cuda-events fp16  # 128MB fp16\n", argv[0]);
        return 0;
    }

    printf("================================================================================\n");
    printf("NCCL AllReduce Benchmark (%s)\n", dtype.name);
    printf("================================================================================\n");
    printf("Data type:    %s (element size: %zu bytes)\n", dtype.name, dtype.elementSize);
    printf("Elements:     %zu (%.2f MB)\n", elemCount, bytes / 1e6);
    printf("Calls:        %d (warmup: %d)\n", numCalls, warmupCalls);
    printf("Timing mode:  %s\n", TimingModeName(timingMode));
    printf("================================================================================\n\n");

    benchmarkNCCL(elemCount, numCalls, warmupCalls, timingMode, dtype);

    return 0;
}
