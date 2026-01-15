// Unit tests for Yali buffer operations
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "../../src/common/buffer_ops.cuh"
#include "../../src/common/validation.cuh"
#include "test_framework.h"

// =============================================================================
// SeedBuffer Tests
// =============================================================================

TEST(SeedBuffer_FP32) {
    if (!yali_test::HasNGPUs(1)) {
        SKIP_TEST("Need at least 1 GPU");
    }
    CUDA_CHECK(cudaSetDevice(0));

    constexpr size_t kCount = 1024;
    float* buf = nullptr;
    float* verify = nullptr;

    CUDA_CHECK(cudaMalloc(&buf, kCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&verify, kCount * sizeof(float)));

    // Seed with test value
    cudaError_t err = yali::SeedBufferSync(buf, kCount, 42.5f);
    EXPECT_EQ(err, cudaSuccess);

    // Convert and verify
    yali::ConvertBufferToFloat(buf, verify, kCount);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> host(kCount);
    CUDA_CHECK(cudaMemcpy(host.data(), verify, kCount * sizeof(float), cudaMemcpyDeviceToHost));

    bool all_match = true;
    for (size_t i = 0; i < kCount; ++i) {
        if (std::fabs(host[i] - 42.5f) > 1e-6f) {
            all_match = false;
            break;
        }
    }
    EXPECT_TRUE(all_match);

    CUDA_CHECK(cudaFree(buf));
    CUDA_CHECK(cudaFree(verify));
}

TEST(SeedBuffer_FP16) {
    if (!yali_test::HasNGPUs(1)) {
        SKIP_TEST("Need at least 1 GPU");
    }
    CUDA_CHECK(cudaSetDevice(0));

    constexpr size_t kCount = 1024;
    __half* buf = nullptr;
    float* verify = nullptr;

    CUDA_CHECK(cudaMalloc(&buf, kCount * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&verify, kCount * sizeof(float)));

    cudaError_t err = yali::SeedBufferSync(buf, kCount, 42.5f);
    EXPECT_EQ(err, cudaSuccess);

    yali::ConvertBufferToFloat(buf, verify, kCount);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> host(kCount);
    CUDA_CHECK(cudaMemcpy(host.data(), verify, kCount * sizeof(float), cudaMemcpyDeviceToHost));

    bool all_match = true;
    for (size_t i = 0; i < kCount; ++i) {
        if (std::fabs(host[i] - 42.5f) > 0.1f) {
            all_match = false;
            break;
        }
    }
    EXPECT_TRUE(all_match);

    CUDA_CHECK(cudaFree(buf));
    CUDA_CHECK(cudaFree(verify));
}

TEST(SeedBuffer_BF16) {
    if (!yali_test::HasNGPUs(1)) {
        SKIP_TEST("Need at least 1 GPU");
    }
    CUDA_CHECK(cudaSetDevice(0));

    constexpr size_t kCount = 1024;
    __nv_bfloat16* buf = nullptr;
    float* verify = nullptr;

    CUDA_CHECK(cudaMalloc(&buf, kCount * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&verify, kCount * sizeof(float)));

    cudaError_t err = yali::SeedBufferSync(buf, kCount, 42.5f);
    EXPECT_EQ(err, cudaSuccess);

    yali::ConvertBufferToFloat(buf, verify, kCount);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> host(kCount);
    CUDA_CHECK(cudaMemcpy(host.data(), verify, kCount * sizeof(float), cudaMemcpyDeviceToHost));

    bool all_match = true;
    for (size_t i = 0; i < kCount; ++i) {
        if (std::fabs(host[i] - 42.5f) > 0.5f) {
            all_match = false;
            break;
        }
    }
    EXPECT_TRUE(all_match);

    CUDA_CHECK(cudaFree(buf));
    CUDA_CHECK(cudaFree(verify));
}

// =============================================================================
// ZeroBuffer Tests
// =============================================================================

TEST(ZeroBuffer_FP32) {
    if (!yali_test::HasNGPUs(1)) {
        SKIP_TEST("Need at least 1 GPU");
    }
    CUDA_CHECK(cudaSetDevice(0));

    constexpr size_t kCount = 1024;
    float* buf = nullptr;

    CUDA_CHECK(cudaMalloc(&buf, kCount * sizeof(float)));

    // First seed with non-zero value
    yali::SeedBufferSync(buf, kCount, 99.0f);

    // Then zero it
    cudaError_t err = yali::ZeroBuffer(buf, kCount);
    EXPECT_EQ(err, cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify zeros
    std::vector<float> host(kCount);
    CUDA_CHECK(cudaMemcpy(host.data(), buf, kCount * sizeof(float), cudaMemcpyDeviceToHost));

    bool all_zero = true;
    for (size_t i = 0; i < kCount; ++i) {
        if (host[i] != 0.0f) {
            all_zero = false;
            break;
        }
    }
    EXPECT_TRUE(all_zero);

    CUDA_CHECK(cudaFree(buf));
}

// =============================================================================
// AllocAndSeed Tests
// =============================================================================

TEST(AllocAndSeed_FP32) {
    if (!yali_test::HasNGPUs(1)) {
        SKIP_TEST("Need at least 1 GPU");
    }
    CUDA_CHECK(cudaSetDevice(0));

    constexpr size_t kCount = 1024;
    float* buf = nullptr;

    cudaError_t err = yali::AllocAndSeed(&buf, kCount, 123.5f);
    EXPECT_EQ(err, cudaSuccess);
    EXPECT_NE(buf, nullptr);

    // Verify value
    std::vector<float> host(kCount);
    CUDA_CHECK(cudaMemcpy(host.data(), buf, kCount * sizeof(float), cudaMemcpyDeviceToHost));

    bool all_match = true;
    for (size_t i = 0; i < kCount; ++i) {
        if (std::fabs(host[i] - 123.5f) > 1e-6f) {
            all_match = false;
            break;
        }
    }
    EXPECT_TRUE(all_match);

    CUDA_CHECK(cudaFree(buf));
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST(SeedBuffer_EmptyCount) {
    if (!yali_test::HasNGPUs(1)) {
        SKIP_TEST("Need at least 1 GPU");
    }
    CUDA_CHECK(cudaSetDevice(0));

    float* buf = nullptr;
    cudaError_t err = yali::SeedBuffer(buf, 0, 42.5f);
    EXPECT_EQ(err, cudaSuccess);  // Should handle gracefully
}

TEST(ZeroBuffer_EmptyCount) {
    if (!yali_test::HasNGPUs(1)) {
        SKIP_TEST("Need at least 1 GPU");
    }
    CUDA_CHECK(cudaSetDevice(0));

    float* buf = nullptr;
    cudaError_t err = yali::ZeroBuffer(buf, static_cast<size_t>(0));
    EXPECT_EQ(err, cudaSuccess);  // Should handle gracefully
}

// =============================================================================
// Main
// =============================================================================

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA device(s)\n", deviceCount);
    return RUN_ALL_TESTS();
}
