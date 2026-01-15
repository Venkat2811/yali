#include <cuda_runtime.h>

#include <chrono>
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

template <typename T>
__global__ void fill_kernel(T* buf, size_t count, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        buf[idx] = static_cast<T>(value);
}

int main() {
    printf("=== Testing Bandwidth Kernel via ops/allreduce.cuh ===\n\n");

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        printf("SKIP: Need 2 GPUs, found %d\n", device_count);
        return 0;
    }

    yali::Comm comm(0, 1);
    if (!comm.ok()) {
        printf("SKIP: P2P not available\n");
        return 0;
    }

    // 128MB = 32M floats (triggers stream kernel at >64MB)
    size_t count = 32 * 1024 * 1024;
    size_t bytes = count * sizeof(float);
    printf("Testing 128MB (%zu floats) - should use stream kernel\n\n", count);

    float *send0, *recv0, *send1, *recv1;

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMalloc(&send0, bytes));
    CHECK_CUDA(cudaMalloc(&recv0, bytes));
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    fill_kernel<<<blocks, threads>>>(send0, count, 1.0f);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMalloc(&send1, bytes));
    CHECK_CUDA(cudaMalloc(&recv1, bytes));
    fill_kernel<<<blocks, threads>>>(send1, count, 2.0f);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("Buffers allocated and seeded (%zu bytes). Running allreduce...\n", bytes);

    cudaError_t err = yali::allreduce(comm, send0, recv0, send1, recv1, count);
    if (err != cudaSuccess) {
        printf("FAIL: allreduce returned %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Allreduce completed. Validating...\n");

    // Validate
    std::vector<float> h0(count), h1(count);
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMemcpy(h0.data(), recv0, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMemcpy(h1.data(), recv1, bytes, cudaMemcpyDeviceToHost));

    int errors0 = 0, errors1 = 0;
    float expected = 3.0f;  // 1.0 + 2.0
    for (size_t i = 0; i < count && errors0 < 10; ++i) {
        if (fabsf(h0[i] - expected) > 1e-5f) {
            if (errors0 == 0)
                printf("GPU0 error at [%zu]: got %.4f, expected %.4f\n", i, h0[i], expected);
            ++errors0;
        }
    }
    for (size_t i = 0; i < count && errors1 < 10; ++i) {
        if (fabsf(h1[i] - expected) > 1e-5f) {
            if (errors1 == 0)
                printf("GPU1 error at [%zu]: got %.4f, expected %.4f\n", i, h1[i], expected);
            ++errors1;
        }
    }

    printf("\nGPU0: %d errors, GPU1: %d errors\n", errors0, errors1);

    // Performance test using wall-clock timing (matches nccl-tests methodology)
    printf("\n--- Performance Test (wall-clock timing, 5 iterations) ---\n");

    // Reset buffers
    CHECK_CUDA(cudaSetDevice(0));
    fill_kernel<<<blocks, threads>>>(send0, count, 1.0f);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaSetDevice(1));
    fill_kernel<<<blocks, threads>>>(send1, count, 2.0f);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Warmup
    for (int i = 0; i < 3; ++i) {
        yali::allreduce(comm, send0, recv0, send1, recv1, count);
    }
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed iterations using wall-clock (like nccl-tests and ThunderKittens)
    const int iters = 5;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        yali::allreduce(comm, send0, recv0, send1, recv1, count);
    }
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / iters;
    double gbps = static_cast<double>(bytes) / (avg_ms * 1e6);

    printf("Bandwidth kernel: %.2f GB/s (%.2f ms per call, wall-clock)\n", gbps, avg_ms);

    cudaSetDevice(0);
    cudaFree(send0);
    cudaFree(recv0);
    cudaSetDevice(1);
    cudaFree(send1);
    cudaFree(recv1);

    bool ok = (errors0 == 0 && errors1 == 0);
    printf("\n=== %s ===\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
