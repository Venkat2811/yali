// Unit tests for Yali data type support (FP32, FP16, BF16)
// Tests type conversion operations and AllReduce correctness across all dtypes
//
// NOTE: AllReduce tests require the kernel code to be refactored into a linkable
// library (Phase 2-4 of the reorganization plan). ValueOps tests work standalone.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "../../src/kernels/type_ops.cuh"
#include "test_framework.h"
#include "yali_launch.h"
#include "yali_tuning.h"

// Seed buffer kernel
template <typename T>
__global__ void seed_buffer_kernel(T* dst, size_t count, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = yali::ValueOps<T>::FromFloat(value);
    }
}

// Convert to float kernel
template <typename T>
__global__ void convert_to_float_kernel(const T* src, float* dst, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = yali::ValueOps<T>::ToFloat(src[idx]);
    }
}

// Add kernel for testing ValueOps::Add
template <typename T>
__global__ void add_kernel(const T* a, const T* b, T* out, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = yali::ValueOps<T>::Add(a[idx], b[idx]);
    }
}

namespace {

template <typename T>
void SeedBuffer(T* dst, size_t count, float value) {
    if (count == 0)
        return;
    constexpr int kThreads = 256;
    int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    seed_buffer_kernel<T><<<blocks, kThreads>>>(dst, count, value);
    cudaDeviceSynchronize();
}

template <typename T>
void ConvertToFloat(const T* src, float* dst, size_t count) {
    if (count == 0)
        return;
    constexpr int kThreads = 256;
    int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    convert_to_float_kernel<T><<<blocks, kThreads>>>(src, dst, count);
    cudaDeviceSynchronize();
}

template <typename T>
void AddBuffers(const T* a, const T* b, T* out, size_t count) {
    if (count == 0)
        return;
    constexpr int kThreads = 256;
    int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    add_kernel<T><<<blocks, kThreads>>>(a, b, out, count);
    cudaDeviceSynchronize();
}

bool VerifyResult(float* host_data, size_t count, float expected, float tolerance) {
    for (size_t i = 0; i < count; ++i) {
        if (std::fabs(host_data[i] - expected) > tolerance) {
            printf("Mismatch at index %zu: got %f, expected %f (diff=%g)\n", i, host_data[i], expected,
                   std::fabs(host_data[i] - expected));
            return false;
        }
    }
    return true;
}

}  // namespace

// =============================================================================
// ValueOps Tests (Type Conversion) - These work standalone
// =============================================================================

TEST(ValueOps_FP32_RoundTrip) {
    if (!yali_test::HasNGPUs(1)) {
        SKIP_TEST("Need at least 1 GPU");
    }

    CUDA_CHECK(cudaSetDevice(0));

    constexpr size_t kCount = 1024;
    float* src = nullptr;
    float* dst = nullptr;

    CUDA_CHECK(cudaMalloc(&src, kCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst, kCount * sizeof(float)));

    // Seed with test value
    SeedBuffer(src, kCount, 42.5f);

    // Convert back to float (should be identity for float)
    ConvertToFloat(src, dst, kCount);

    std::vector<float> host(kCount);
    CUDA_CHECK(cudaMemcpy(host.data(), dst, kCount * sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_TRUE(VerifyResult(host.data(), kCount, 42.5f, 1e-6f));

    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dst));
}

TEST(ValueOps_FP16_RoundTrip) {
    if (!yali_test::HasNGPUs(1)) {
        SKIP_TEST("Need at least 1 GPU");
    }

    CUDA_CHECK(cudaSetDevice(0));

    constexpr size_t kCount = 1024;
    __half* src = nullptr;
    float* dst = nullptr;

    CUDA_CHECK(cudaMalloc(&src, kCount * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dst, kCount * sizeof(float)));

    // Seed with test value
    SeedBuffer(src, kCount, 42.5f);

    // Convert back to float
    ConvertToFloat(src, dst, kCount);

    std::vector<float> host(kCount);
    CUDA_CHECK(cudaMemcpy(host.data(), dst, kCount * sizeof(float), cudaMemcpyDeviceToHost));

    // FP16 has less precision, use larger tolerance
    EXPECT_TRUE(VerifyResult(host.data(), kCount, 42.5f, 0.1f));

    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dst));
}

TEST(ValueOps_BF16_RoundTrip) {
    if (!yali_test::HasNGPUs(1)) {
        SKIP_TEST("Need at least 1 GPU");
    }

    CUDA_CHECK(cudaSetDevice(0));

    constexpr size_t kCount = 1024;
    __nv_bfloat16* src = nullptr;
    float* dst = nullptr;

    CUDA_CHECK(cudaMalloc(&src, kCount * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dst, kCount * sizeof(float)));

    // Seed with test value
    SeedBuffer(src, kCount, 42.5f);

    // Convert back to float
    ConvertToFloat(src, dst, kCount);

    std::vector<float> host(kCount);
    CUDA_CHECK(cudaMemcpy(host.data(), dst, kCount * sizeof(float), cudaMemcpyDeviceToHost));

    // BF16 has less precision than FP32, use appropriate tolerance
    EXPECT_TRUE(VerifyResult(host.data(), kCount, 42.5f, 0.5f));

    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dst));
}

// =============================================================================
// ValueOps Add Tests - These work standalone
// =============================================================================

TEST(ValueOps_FP32_Add) {
    if (!yali_test::HasNGPUs(1)) {
        SKIP_TEST("Need at least 1 GPU");
    }

    CUDA_CHECK(cudaSetDevice(0));

    constexpr size_t kCount = 1024;
    float *a = nullptr, *b = nullptr, *out = nullptr;
    float* dst = nullptr;

    CUDA_CHECK(cudaMalloc(&a, kCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, kCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out, kCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst, kCount * sizeof(float)));

    SeedBuffer(a, kCount, 1.5f);
    SeedBuffer(b, kCount, 2.5f);

    AddBuffers(a, b, out, kCount);
    ConvertToFloat(out, dst, kCount);

    std::vector<float> host(kCount);
    CUDA_CHECK(cudaMemcpy(host.data(), dst, kCount * sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_TRUE(VerifyResult(host.data(), kCount, 4.0f, 1e-6f));

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(dst));
}

TEST(ValueOps_FP16_Add) {
    if (!yali_test::HasNGPUs(1)) {
        SKIP_TEST("Need at least 1 GPU");
    }

    CUDA_CHECK(cudaSetDevice(0));

    constexpr size_t kCount = 1024;
    __half *a = nullptr, *b = nullptr, *out = nullptr;
    float* dst = nullptr;

    CUDA_CHECK(cudaMalloc(&a, kCount * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&b, kCount * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&out, kCount * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dst, kCount * sizeof(float)));

    SeedBuffer(a, kCount, 1.5f);
    SeedBuffer(b, kCount, 2.5f);

    AddBuffers(a, b, out, kCount);
    ConvertToFloat(out, dst, kCount);

    std::vector<float> host(kCount);
    CUDA_CHECK(cudaMemcpy(host.data(), dst, kCount * sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_TRUE(VerifyResult(host.data(), kCount, 4.0f, 0.1f));

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(dst));
}

TEST(ValueOps_BF16_Add) {
    if (!yali_test::HasNGPUs(1)) {
        SKIP_TEST("Need at least 1 GPU");
    }

    CUDA_CHECK(cudaSetDevice(0));

    constexpr size_t kCount = 1024;
    __nv_bfloat16 *a = nullptr, *b = nullptr, *out = nullptr;
    float* dst = nullptr;

    CUDA_CHECK(cudaMalloc(&a, kCount * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&b, kCount * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&out, kCount * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dst, kCount * sizeof(float)));

    SeedBuffer(a, kCount, 1.5f);
    SeedBuffer(b, kCount, 2.5f);

    AddBuffers(a, b, out, kCount);
    ConvertToFloat(out, dst, kCount);

    std::vector<float> host(kCount);
    CUDA_CHECK(cudaMemcpy(host.data(), dst, kCount * sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_TRUE(VerifyResult(host.data(), kCount, 4.0f, 0.5f));

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(dst));
}

// =============================================================================
// Element Size Tests
// =============================================================================

TEST(ElementSize_FP32) {
    EXPECT_EQ(sizeof(float), 4u);
}

TEST(ElementSize_FP16) {
    EXPECT_EQ(sizeof(__half), 2u);
}

TEST(ElementSize_BF16) {
    EXPECT_EQ(sizeof(__nv_bfloat16), 2u);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA device(s)\n", deviceCount);

    if (deviceCount >= 2) {
        int canAccess01 = 0, canAccess10 = 0;
        cudaDeviceCanAccessPeer(&canAccess01, 0, 1);
        cudaDeviceCanAccessPeer(&canAccess10, 1, 0);
        printf("P2P access: GPU0->GPU1=%d, GPU1->GPU0=%d\n", canAccess01, canAccess10);
    }

    return RUN_ALL_TESTS();
}
