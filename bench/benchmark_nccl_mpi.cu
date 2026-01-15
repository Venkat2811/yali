/**
 * AllReduce Benchmark (NCCL + MPI)
 *
 * Multi-process version of benchmark_nccl.cu.
 * Each MPI rank manages one GPU.
 *
 * Usage:
 *   mpirun -np 2 ./benchmark_nccl_mpi <elements> <calls> [timing]
 *   timing: throughput (default), latency, cuda-events
 *
 * This benchmark mimics how inference engines actually use AllReduce:
 * 1. Setup communicators/buffers once
 * 2. Run N allreduce calls in a tight loop (no sync between)
 * 3. Sync only at end
 * 4. Measure total throughput
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <nccl.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <string>

#define CHECK_CUDA(cmd)                                                                                                \
    do {                                                                                                               \
        cudaError_t e = cmd;                                                                                           \
        if (e != cudaSuccess) {                                                                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                      \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                              \
        }                                                                                                              \
    } while (0)

#define CHECK_NCCL(cmd)                                                                                                \
    do {                                                                                                               \
        ncclResult_t r = cmd;                                                                                          \
        if (r != ncclSuccess) {                                                                                        \
            fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r));                      \
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

//------------------------------------------------------------------------------
// Data type configuration
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
    std::string lowered = dtypeStr ? std::string(dtypeStr) : std::string("fp32");
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

void benchmarkNCCL(int rank, int worldSize, size_t elemCount, int numCalls, int warmupCalls,
                   const NCCLDTypeConfig& dtype, TimingMode timingMode) {
    // Each rank owns one GPU
    CHECK_CUDA(cudaSetDevice(rank));
    const size_t bytes = elemCount * dtype.elementSize;

    // Setup buffers (per-rank)
    void* sendbuff = nullptr;
    void* recvbuff = nullptr;
    CHECK_CUDA(cudaMalloc(&sendbuff, bytes));
    CHECK_CUDA(cudaMalloc(&recvbuff, bytes));

    // Create stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // NCCL setup: rank 0 creates unique ID, broadcasts to others
    ncclUniqueId ncclId;
    if (rank == 0) {
        CHECK_NCCL(ncclGetUniqueId(&ncclId));
    }
    CHECK_MPI(MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD));

    // Initialize NCCL communicator (per-rank)
    ncclComm_t comm;
    CHECK_NCCL(ncclCommInitRank(&comm, worldSize, ncclId, rank));

    if (rank == 0) {
        printf("Data type: %s\n", dtype.name);
        printf("Timing mode: %s\n", TimingModeName(timingMode));
    }

    // Lambda for launching one AllReduce
    auto launchAllReduce = [&]() {
        CHECK_NCCL(ncclAllReduce(sendbuff, recvbuff, elemCount, dtype.ncclType, ncclSum, comm, stream));
    };

    // Sync this rank's stream
    auto syncStream = [&]() { CHECK_CUDA(cudaStreamSynchronize(stream)); };

    // Warmup - like real inference warmup
    for (int iter = 0; iter < warmupCalls; iter++) {
        launchAllReduce();
    }
    syncStream();
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

    // Timed run - depends on timing mode
    double localMs = 0.0;

    if (timingMode == TimingMode::CudaEvents) {
        // CUDA events around batch (ThunderKittens methodology)
        cudaEvent_t startEvent, stopEvent;
        CHECK_CUDA(cudaEventCreate(&startEvent));
        CHECK_CUDA(cudaEventCreate(&stopEvent));

        // Pre-barrier to ensure all GPUs are idle
        syncStream();
        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

        // Record start
        CHECK_CUDA(cudaEventRecord(startEvent, stream));

        // Fire all iterations
        for (int iter = 0; iter < numCalls; iter++) {
            launchAllReduce();
        }

        // Record stop and sync
        CHECK_CUDA(cudaEventRecord(stopEvent, stream));
        syncStream();

        float elapsedMs = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
        localMs = static_cast<double>(elapsedMs);

        CHECK_CUDA(cudaEventDestroy(startEvent));
        CHECK_CUDA(cudaEventDestroy(stopEvent));

    } else if (timingMode == TimingMode::Throughput) {
        // Wall-clock, fire-and-forget, single sync at end
        syncStream();
        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

        auto start = std::chrono::steady_clock::now();

        for (int iter = 0; iter < numCalls; iter++) {
            launchAllReduce();
        }
        syncStream();

        auto end = std::chrono::steady_clock::now();
        localMs = std::chrono::duration<double, std::milli>(end - start).count();

    } else {
        // Latency mode: sync after each iteration
        syncStream();
        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

        auto start = std::chrono::steady_clock::now();

        for (int iter = 0; iter < numCalls; iter++) {
            launchAllReduce();
            syncStream();
        }

        auto end = std::chrono::steady_clock::now();
        localMs = std::chrono::duration<double, std::milli>(end - start).count();
    }

    // MPI barrier after measurement
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

    // Get global max time across all ranks
    double globalMaxMs = 0.0;
    CHECK_MPI(MPI_Allreduce(&localMs, &globalMaxMs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

    double avgUs = (globalMaxMs * 1000.0) / numCalls;

    // NCCL busBw formula for AllReduce: data_size * 2 * (nranks-1) / nranks / time
    double dataBytes = static_cast<double>(bytes);
    double busBwFactor = 2.0 * static_cast<double>(worldSize - 1) / static_cast<double>(worldSize);
    double gbps = (dataBytes * busBwFactor / 1e9) / (avgUs / 1e6);
    double solPercent = gbps / 100.0 * 100.0;  // vs 100 GB/s unidirectional NVLink

    // Print results (rank 0 only)
    if (rank == 0) {
        const char* modeStr = (timingMode == TimingMode::CudaEvents)   ? "cuda-events"
                              : (timingMode == TimingMode::Throughput) ? "throughput"
                                                                       : "latency";
        printf("NCCL MPI (%s, %s): %d calls, %.2f us/call avg, %.2f GB/s (%.1f%% SoL)\n", dtype.name, modeStr, numCalls,
               avgUs, gbps, solPercent);
    }

    // Cleanup
    ncclCommDestroy(comm);
    cudaFree(sendbuff);
    cudaFree(recvbuff);
    cudaStreamDestroy(stream);
}

int main(int argc, char** argv) {
    // Initialize MPI with thread support
    int provided;
    CHECK_MPI(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));
    if (provided < MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "MPI does not provide MPI_THREAD_MULTIPLE support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, worldSize;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));

    if (worldSize != 2) {
        if (rank == 0) {
            fprintf(stderr, "Error: This benchmark requires exactly 2 MPI ranks (got %d)\n", worldSize);
        }
        MPI_Finalize();
        return 1;
    }

    // Set GPU for this rank
    CHECK_CUDA(cudaSetDevice(rank));

    // Parse arguments
    size_t elemCount = 262144;  // 1MB
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

    // Print header (rank 0 only)
    if (rank == 0) {
        printf("================================================================================\n");
        printf("NCCL AllReduce Benchmark (MPI, %d ranks, %s)\n", worldSize, dtype.name);
        printf("================================================================================\n");
        printf("Data type:    %s (element size: %zu bytes)\n", dtype.name, dtype.elementSize);
        printf("Elements:     %zu (%.2f MB)\n", elemCount, static_cast<double>(bytes) / 1e6);
        printf("Calls:        %d (warmup: %d)\n", numCalls, warmupCalls);
        printf("Timing mode:  %s\n", TimingModeName(timingMode));
        printf("================================================================================\n\n");
    }

    benchmarkNCCL(rank, worldSize, elemCount, numCalls, warmupCalls, dtype, timingMode);

    if (rank == 0) {
        printf("\nUsage: mpirun -np 2 %s <elements> <calls> [timing] [dtype]\n", argv[0]);
        printf("  timing: throughput (default), latency, cuda-events\n");
        printf("  dtype: fp32 (default), fp16, bf16\n");
    }

    MPI_Finalize();
    return 0;
}
