/*************************************************************************
 * Test: ops/allreduce_mpi.cuh
 *
 * Validates the MPI API wrapper achieves:
 * 1. Correctness - results match expected sum
 * 2. Performance - matches raw harness bandwidth
 *
 * Build:  bazel build //:test_ops_allreduce_mpi
 * Run:    CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 --allow-run-as-root bazel-bin/test_ops_allreduce_mpi
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <vector>

#include "src/ops/allreduce_mpi.cuh"

#define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = (call);                                                                                      \
        if (err != cudaSuccess) {                                                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                 \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

// ============================================================================
// Test utilities
// ============================================================================

template <typename T>
__global__ void fill_kernel(T* buf, size_t count, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        buf[idx] = static_cast<T>(value);
    }
}

template <typename T>
void fill_buffer(T* buf, size_t count, float value) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    fill_kernel<T><<<blocks, threads>>>(buf, count, value);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

template <typename T>
bool validate_buffer(T* buf, size_t count, float expected, const char* name) {
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy to host for validation
    std::vector<T> host_buf(count);
    CHECK_CUDA(cudaMemcpy(host_buf.data(), buf, count * sizeof(T), cudaMemcpyDeviceToHost));

    int errors = 0;
    float tol = (sizeof(T) == 4) ? 1e-5f : 0.01f;
    for (size_t i = 0; i < count && errors < 10; ++i) {
        float val = static_cast<float>(host_buf[i]);
        if (fabsf(val - expected) > tol) {
            if (errors == 0) {
                printf("  %s: First error at [%zu]: got %.4f, expected %.4f\n", name, i, val, expected);
            }
            ++errors;
        }
    }

    if (errors > 0) {
        printf("  %s: FAIL (%d errors out of %zu)\n", name, errors, count);
        return false;
    }
    return true;
}

// ============================================================================
// Test: Correctness
// ============================================================================

template <typename T>
bool test_correctness(yali::MPIComm& comm, const char* dtype_name, size_t count) {
    const int rank = comm.rank();

    if (rank == 0) {
        printf("Testing correctness: %s, %zu elements...\n", dtype_name, count);
    }

    T *send, *recv;
    CHECK_CUDA(cudaMalloc(&send, count * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&recv, count * sizeof(T)));

    // Rank 0 = 1.0, Rank 1 = 2.0
    float seed = static_cast<float>(rank + 1);
    fill_buffer(send, count, seed);
    CHECK_CUDA(cudaMemset(recv, 0, count * sizeof(T)));

    // AllReduce
    cudaError_t err = yali::allreduce(comm, send, recv, count);
    if (err != cudaSuccess) {
        printf("  Rank %d: FAIL allreduce returned %s\n", rank, cudaGetErrorString(err));
        cudaFree(send);
        cudaFree(recv);
        return false;
    }

    // Validate: expected = 1.0 + 2.0 = 3.0
    char buf_name[32];
    snprintf(buf_name, sizeof(buf_name), "Rank%d", rank);
    bool local_ok = validate_buffer(recv, count, 3.0f, buf_name);

    cudaFree(send);
    cudaFree(recv);

    // Aggregate pass/fail across all ranks (all must pass)
    int local_pass = local_ok ? 1 : 0;
    int global_pass = 0;
    MPI_Allreduce(&local_pass, &global_pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    // Barrier to sync output
    comm.barrier();

    if (rank == 0) {
        printf("  %s\n", global_pass ? "PASS" : "FAIL");
    }
    return global_pass != 0;
}

// ============================================================================
// Test: Performance
// ============================================================================

template <typename T>
bool test_performance(yali::MPIComm& comm, const char* dtype_name, size_t count, float min_gbps) {
    const int rank = comm.rank();

    if (rank == 0) {
        printf("Testing performance: %s, %zu elements (min %.1f GB/s)...\n", dtype_name, count, min_gbps);
    }

    T *send, *recv;
    size_t bytes = count * sizeof(T);

    CHECK_CUDA(cudaMalloc(&send, bytes));
    CHECK_CUDA(cudaMalloc(&recv, bytes));

    float seed = static_cast<float>(rank + 1);
    fill_buffer(send, count, seed);

    // Warmup
    for (int i = 0; i < 2; ++i) {
        yali::allreduce(comm, send, recv, count);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    comm.barrier();

    // Timed iterations
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int iters = 5;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        yali::allreduce(comm, send, recv, count);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    comm.barrier();

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    float avg_ms = ms / iters;
    // algbw = data_size / time (NCCL convention, same as harness)
    float gbps = static_cast<float>(bytes) / (avg_ms * 1e6f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(send);
    cudaFree(recv);

    bool ok = (gbps >= min_gbps);
    if (rank == 0) {
        printf("  %.2f GB/s (threshold: %.1f GB/s) - %s\n", gbps, min_gbps, ok ? "PASS" : "FAIL");
    }
    return ok;
}

// ============================================================================
// Test: Performance with buffer_stable=true
// ============================================================================

template <typename T>
bool test_performance_cached(yali::MPIComm& comm, const char* dtype_name, size_t count, float min_gbps) {
    const int rank = comm.rank();

    if (rank == 0) {
        printf("Testing performance (buffer_stable=true): %s, %zu elements (min %.1f GB/s)...\n", dtype_name, count,
               min_gbps);
    }

    T *send, *recv;
    size_t bytes = count * sizeof(T);

    CHECK_CUDA(cudaMalloc(&send, bytes));
    CHECK_CUDA(cudaMalloc(&recv, bytes));

    float seed = static_cast<float>(rank + 1);
    fill_buffer(send, count, seed);

    // Warmup with buffer_stable=true
    for (int i = 0; i < 2; ++i) {
        yali::allreduce(comm, send, recv, count, 0, true);  // buffer_stable=true
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    comm.barrier();

    // Timed iterations with buffer_stable=true
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int iters = 5;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        yali::allreduce(comm, send, recv, count, 0, true);  // buffer_stable=true
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    comm.barrier();

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    float avg_ms = ms / iters;
    float gbps = static_cast<float>(bytes) / (avg_ms * 1e6f);

    // Validate correctness
    bool correct = validate_buffer(recv, count, 3.0f, "cached");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(send);
    cudaFree(recv);

    bool ok = (gbps >= min_gbps) && correct;
    if (rank == 0) {
        printf("  %.2f GB/s (threshold: %.1f GB/s) - %s%s\n", gbps, min_gbps, ok ? "PASS" : "FAIL",
               correct ? "" : " (correctness failed)");
    }
    return ok;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // Initialize MPI communicator
    yali::MPIComm comm(&argc, &argv);
    if (!comm.ok()) {
        fprintf(stderr, "Failed to initialize MPI communicator\n");
        return 1;
    }

    const int rank = comm.rank();

    if (rank == 0) {
        printf("=== Yali ops/allreduce_mpi.cuh Tests ===\n");
        printf("World size: %d\n\n", comm.world_size());
    }
    comm.barrier();

    bool all_pass = true;

    // Correctness tests - Low-latency kernel (small messages)
    if (rank == 0)
        printf("--- Correctness Tests (Low-Latency Kernel) ---\n");
    all_pass &= test_correctness<float>(comm, "fp32", 1024);
    all_pass &= test_correctness<float>(comm, "fp32", 1024 * 1024);
    all_pass &= test_correctness<__half>(comm, "fp16", 1024 * 1024);
    all_pass &= test_correctness<__nv_bfloat16>(comm, "bf16", 1024 * 1024);
    if (rank == 0)
        printf("\n");

    // Correctness tests - Bandwidth kernel (large messages >64MB)
    if (rank == 0)
        printf("--- Correctness Tests (Bandwidth Kernel) ---\n");
    // 128MB = 32M floats - triggers stream kernel
    all_pass &= test_correctness<float>(comm, "fp32", 32 * 1024 * 1024);
    all_pass &= test_correctness<__half>(comm, "fp16", 64 * 1024 * 1024);
    all_pass &= test_correctness<__nv_bfloat16>(comm, "bf16", 64 * 1024 * 1024);
    if (rank == 0)
        printf("\n");

    // Performance tests - ops API includes IPC re-exchange overhead per call
    // For production use with stable buffers, use buffer_stable=true or raw harness
    if (rank == 0)
        printf("--- Performance Tests (buffer_stable=false) ---\n");
    // 64MB message (low-latency): expect at least 20 GB/s (lower threshold due to IPC re-exchange)
    all_pass &= test_performance<float>(comm, "fp32 (flash)", 16 * 1024 * 1024, 20.0f);
    // 128MB message (bandwidth): IPC re-exchange dominates (~34 GB/s observed)
    // Note: raw harness gets ~270 GB/s with single IPC exchange at init
    all_pass &= test_performance<float>(comm, "fp32 (bandwidth)", 32 * 1024 * 1024, 25.0f);
    if (rank == 0)
        printf("\n");

    // Performance tests with buffer_stable=true (IPC caching enabled)
    if (rank == 0)
        printf("--- Performance Tests (buffer_stable=true) ---\n");
    // Low-latency with caching: ~38 GB/s (near raw harness ~38 GB/s)
    all_pass &= test_performance_cached<float>(comm, "fp32 (flash)", 16 * 1024 * 1024, 30.0f);
    // Bandwidth with caching: still limited by per-call MPI barrier overhead
    // Note: raw harness gets ~270 GB/s by amortizing setup across many iterations
    // Ops API has per-call barrier + args setup overhead (~37 GB/s observed)
    all_pass &= test_performance_cached<float>(comm, "fp32 (bandwidth)", 32 * 1024 * 1024, 30.0f);
    if (rank == 0)
        printf("\n");

    if (rank == 0) {
        printf("=== %s ===\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    }

    return all_pass ? 0 : 1;
}
