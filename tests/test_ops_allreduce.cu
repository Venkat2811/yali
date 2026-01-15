/*************************************************************************
 * Test: ops/allreduce.cuh
 *
 * Validates the simple API wrapper achieves:
 * 1. Correctness - results match expected sum
 * 2. Performance - matches raw harness bandwidth
 *
 * Build:  bazel build //:test_ops_allreduce
 * Run:    CUDA_VISIBLE_DEVICES=0,1 bazel-bin/test_ops_allreduce
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "src/ops/allreduce.cuh"

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
void fill_buffer(T* buf, size_t count, float value, int device) {
    CHECK_CUDA(cudaSetDevice(device));
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    fill_kernel<T><<<blocks, threads>>>(buf, count, value);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

template <typename T>
__global__ void check_kernel(const T* buf, size_t count, float expected, int* errors) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float val = static_cast<float>(buf[idx]);
        float diff = fabsf(val - expected);
        float tol = (sizeof(T) == 4) ? 1e-5f : 0.01f;  // FP16/BF16 need more tolerance
        if (diff > tol) {
            atomicAdd(errors, 1);
        }
    }
}

template <typename T>
bool validate_buffer(T* buf, size_t count, float expected, const char* name, int device) {
    CHECK_CUDA(cudaSetDevice(device));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy to host for validation (simpler and works correctly across GPUs)
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
bool test_correctness(const char* dtype_name, size_t count) {
    printf("Testing correctness: %s, %zu elements...\n", dtype_name, count);

    yali::Comm comm(0, 1);
    if (!comm.ok()) {
        printf("  SKIP: P2P not available\n");
        return true;
    }

    T *send0, *recv0, *send1, *recv1;

    // Allocate separate send/recv buffers (required by kernel - not in-place)
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMalloc(&send0, count * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&recv0, count * sizeof(T)));
    fill_buffer(send0, count, 1.0f, 0);

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMalloc(&send1, count * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&recv1, count * sizeof(T)));
    fill_buffer(send1, count, 2.0f, 1);

    // AllReduce with separate send/recv
    cudaError_t err = yali::allreduce(comm, send0, recv0, send1, recv1, count);
    if (err != cudaSuccess) {
        printf("  FAIL: allreduce returned %s\n", cudaGetErrorString(err));
        cudaSetDevice(0);
        cudaFree(send0);
        cudaFree(recv0);
        cudaSetDevice(1);
        cudaFree(send1);
        cudaFree(recv1);
        return false;
    }

    // Validate: expected = 1.0 + 2.0 = 3.0
    bool ok = true;
    ok &= validate_buffer(recv0, count, 3.0f, "GPU0", 0);
    ok &= validate_buffer(recv1, count, 3.0f, "GPU1", 1);

    cudaSetDevice(0);
    cudaFree(send0);
    cudaFree(recv0);
    cudaSetDevice(1);
    cudaFree(send1);
    cudaFree(recv1);

    printf("  %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ============================================================================
// Test: Performance
// ============================================================================

template <typename T>
bool test_performance(const char* dtype_name, size_t count, float min_gbps) {
    printf("Testing performance: %s, %zu elements (min %.1f GB/s)...\n", dtype_name, count, min_gbps);

    yali::Comm comm(0, 1);
    if (!comm.ok()) {
        printf("  SKIP: P2P not available\n");
        return true;
    }

    T *send0, *recv0, *send1, *recv1;
    size_t bytes = count * sizeof(T);

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMalloc(&send0, bytes));
    CHECK_CUDA(cudaMalloc(&recv0, bytes));
    fill_buffer(send0, count, 1.0f, 0);

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMalloc(&send1, bytes));
    CHECK_CUDA(cudaMalloc(&recv1, bytes));
    fill_buffer(send1, count, 2.0f, 1);

    // Warmup
    for (int i = 0; i < 2; ++i) {
        yali::allreduce(comm, send0, recv0, send1, recv1, count);
    }
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed iterations
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int iters = 5;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        yali::allreduce(comm, send0, recv0, send1, recv1, count);
    }
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    float avg_ms = ms / iters;
    // algbw = data_size / time (NCCL convention, same as harness)
    float gbps = static_cast<float>(bytes) / (avg_ms * 1e6f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaSetDevice(0);
    cudaFree(send0);
    cudaFree(recv0);
    cudaSetDevice(1);
    cudaFree(send1);
    cudaFree(recv1);

    bool ok = (gbps >= min_gbps);
    printf("  %.2f GB/s (threshold: %.1f GB/s) - %s\n", gbps, min_gbps, ok ? "PASS" : "FAIL");
    return ok;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("=== Yali ops/allreduce.cuh Tests ===\n\n");

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        printf("SKIP: Need 2 GPUs, found %d\n", device_count);
        return 0;
    }

    bool all_pass = true;

    // Correctness tests (various sizes and dtypes)
    printf("--- Correctness Tests ---\n");
    all_pass &= test_correctness<float>("fp32", 1024);
    all_pass &= test_correctness<float>("fp32", 1024 * 1024);
    all_pass &= test_correctness<__half>("fp16", 1024 * 1024);
    all_pass &= test_correctness<__nv_bfloat16>("bf16", 1024 * 1024);
    printf("\n");

    // Performance tests - ops API should match raw harness performance
    printf("--- Performance Tests ---\n");
    // 64MB message: expect at least 30 GB/s with low-latency kernel
    // Peak stream kernel (>64MB) gets ~260 GB/s but low-latency ~38 GB/s
    all_pass &= test_performance<float>("fp32", 16 * 1024 * 1024, 30.0f);
    printf("\n");

    printf("=== %s ===\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
