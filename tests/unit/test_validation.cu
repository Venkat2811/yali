// Unit tests for Yali validation utilities
#include <cuda_runtime.h>

#include <vector>

#include "../../src/common/validation.cuh"
#include "test_framework.h"

// =============================================================================
// VerifyAllReduceSum tests (pure host-side logic)
// =============================================================================

TEST(VerifyAllReduceSum_AllCorrect) {
    std::vector<float> data(1024, 3.0f);
    auto result = yali::VerifyAllReduceSum(data.data(), data.size(), 3.0f);
    EXPECT_TRUE(result.passed);
    EXPECT_EQ(result.mismatches, 0u);
}

TEST(VerifyAllReduceSum_FirstMismatch) {
    std::vector<float> data(1024, 3.0f);
    data[100] = 4.0f;  // Inject mismatch
    auto result = yali::VerifyAllReduceSum(data.data(), data.size(), 3.0f);
    EXPECT_FALSE(result.passed);
    EXPECT_EQ(result.first_mismatch_idx, 100u);
    EXPECT_EQ(result.mismatches, 1u);
}

TEST(VerifyAllReduceSum_MultipleMismatches) {
    std::vector<float> data(1024, 3.0f);
    data[50] = 5.0f;
    data[200] = 0.0f;
    data[500] = 10.0f;
    auto result = yali::VerifyAllReduceSum(data.data(), data.size(), 3.0f);
    EXPECT_FALSE(result.passed);
    EXPECT_EQ(result.first_mismatch_idx, 50u);
    EXPECT_EQ(result.mismatches, 3u);
}

TEST(VerifyAllReduceSum_Tolerance) {
    std::vector<float> data(100, 3.0f);
    data[10] = 3.0005f;  // Within tolerance
    auto result = yali::VerifyAllReduceSum(data.data(), data.size(), 3.0f, 0.001f);
    EXPECT_TRUE(result.passed);
}

TEST(VerifyAllReduceSum_OutsideTolerance) {
    std::vector<float> data(100, 3.0f);
    data[10] = 3.002f;  // Outside tolerance
    auto result = yali::VerifyAllReduceSum(data.data(), data.size(), 3.0f, 0.001f);
    EXPECT_FALSE(result.passed);
}

TEST(VerifyAllReduceSum_EmptyData) {
    std::vector<float> data;
    auto result = yali::VerifyAllReduceSum(data.data(), 0, 3.0f);
    EXPECT_TRUE(result.passed);
    EXPECT_EQ(result.total_checked, 0u);
}

// =============================================================================
// ConvertBufferToFloat tests (requires GPU)
// =============================================================================

TEST(ConvertBufferToFloat_FP32) {
    if (!yali_test::HasNGPUs(1)) {
        SKIP_TEST("Need at least 1 GPU");
    }
    CUDA_CHECK(cudaSetDevice(0));

    constexpr size_t kCount = 1024;
    float* src = nullptr;
    float* dst = nullptr;

    CUDA_CHECK(cudaMalloc(&src, kCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst, kCount * sizeof(float)));

    // Fill with known value
    std::vector<float> host_src(kCount, 42.5f);
    CUDA_CHECK(cudaMemcpy(src, host_src.data(), kCount * sizeof(float), cudaMemcpyHostToDevice));

    // Convert (identity for float)
    yali::ConvertBufferToFloat(src, dst, kCount);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify
    std::vector<float> host_dst(kCount);
    CUDA_CHECK(cudaMemcpy(host_dst.data(), dst, kCount * sizeof(float), cudaMemcpyDeviceToHost));

    bool all_match = true;
    for (size_t i = 0; i < kCount; ++i) {
        if (std::fabs(host_dst[i] - 42.5f) > 1e-6f) {
            all_match = false;
            break;
        }
    }
    EXPECT_TRUE(all_match);

    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dst));
}

TEST(ConvertBufferToFloat_FP16) {
    if (!yali_test::HasNGPUs(1)) {
        SKIP_TEST("Need at least 1 GPU");
    }
    CUDA_CHECK(cudaSetDevice(0));

    constexpr size_t kCount = 1024;
    __half* src = nullptr;
    float* dst = nullptr;

    CUDA_CHECK(cudaMalloc(&src, kCount * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dst, kCount * sizeof(float)));

    // Fill with known value (convert on host)
    std::vector<__half> host_src(kCount);
    for (size_t i = 0; i < kCount; ++i) {
        host_src[i] = __float2half(42.5f);
    }
    CUDA_CHECK(cudaMemcpy(src, host_src.data(), kCount * sizeof(__half), cudaMemcpyHostToDevice));

    // Convert
    yali::ConvertBufferToFloat(src, dst, kCount);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify
    std::vector<float> host_dst(kCount);
    CUDA_CHECK(cudaMemcpy(host_dst.data(), dst, kCount * sizeof(float), cudaMemcpyDeviceToHost));

    bool all_match = true;
    for (size_t i = 0; i < kCount; ++i) {
        if (std::fabs(host_dst[i] - 42.5f) > 0.1f) {
            all_match = false;
            break;
        }
    }
    EXPECT_TRUE(all_match);

    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dst));
}

TEST(ConvertBufferToFloat_BF16) {
    if (!yali_test::HasNGPUs(1)) {
        SKIP_TEST("Need at least 1 GPU");
    }
    CUDA_CHECK(cudaSetDevice(0));

    constexpr size_t kCount = 1024;
    __nv_bfloat16* src = nullptr;
    float* dst = nullptr;

    CUDA_CHECK(cudaMalloc(&src, kCount * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dst, kCount * sizeof(float)));

    // Fill with known value (convert on host)
    std::vector<__nv_bfloat16> host_src(kCount);
    for (size_t i = 0; i < kCount; ++i) {
        host_src[i] = __float2bfloat16(42.5f);
    }
    CUDA_CHECK(cudaMemcpy(src, host_src.data(), kCount * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // Convert
    yali::ConvertBufferToFloat(src, dst, kCount);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify
    std::vector<float> host_dst(kCount);
    CUDA_CHECK(cudaMemcpy(host_dst.data(), dst, kCount * sizeof(float), cudaMemcpyDeviceToHost));

    bool all_match = true;
    for (size_t i = 0; i < kCount; ++i) {
        if (std::fabs(host_dst[i] - 42.5f) > 0.5f) {  // BF16 has lower precision
            all_match = false;
            break;
        }
    }
    EXPECT_TRUE(all_match);

    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dst));
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
