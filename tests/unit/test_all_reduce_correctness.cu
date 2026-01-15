// Unit tests for Yali AllReduce correctness
// Tests basic AllReduce functionality across different sizes and configurations
//
// NOTE: These tests require the kernel code to be refactored into a linkable
// library (Phase 2-4 of the reorganization plan). Until then, use the shell-based
// tests in run_unit_tests.sh for quick validation.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "../../src/kernels/type_ops.cuh"
#include "test_framework.h"
#include "yali_launch.h"
#include "yali_tuning.h"

// Forward declarations - will link once kernels are extracted to library
template <typename T, int PrefetchStages>
__global__ void yaliFlashKernel(const YaliLaunchArgs* argsArray, int laneCount, int ctasPerLane);

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

namespace {

// Helper to seed a GPU buffer
template <typename T>
void SeedBuffer(T* dst, size_t count, float value) {
    if (count == 0)
        return;
    constexpr int kThreads = 256;
    int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    seed_buffer_kernel<T><<<blocks, kThreads>>>(dst, count, value);
    cudaDeviceSynchronize();
}

// Helper to convert buffer to float for verification
template <typename T>
void ConvertToFloat(const T* src, float* dst, size_t count) {
    if (count == 0)
        return;
    constexpr int kThreads = 256;
    int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    convert_to_float_kernel<T><<<blocks, kThreads>>>(src, dst, count);
    cudaDeviceSynchronize();
}

// Enable P2P access
void EnableP2P(int dev0, int dev1) {
    int canAccess = 0;
    cudaDeviceCanAccessPeer(&canAccess, dev0, dev1);
    if (!canAccess)
        return;

    cudaSetDevice(dev0);
    cudaError_t err = cudaDeviceEnablePeerAccess(dev1, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
    }

    cudaSetDevice(dev1);
    err = cudaDeviceEnablePeerAccess(dev0, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
    }
}

// Verify AllReduce result (sum of rank values should be 1+2=3 for 2 ranks)
// NOTE: Will be used once kernel tests are enabled after Phase 4 refactoring
#if 0  // Disabled until kernel tests are enabled
bool VerifyAllReduceResult(float* host_data, size_t count, float expected, float tolerance = 1e-3f) {
  for (size_t i = 0; i < count; ++i) {
    if (std::fabs(host_data[i] - expected) > tolerance) {
      printf("Mismatch at index %zu: got %f, expected %f\n", i, host_data[i], expected);
      return false;
    }
  }
  return true;
}
#endif

}  // namespace

// =============================================================================
// Environment Tests
// =============================================================================

TEST(HasTwoGPUs) {
    if (!yali_test::HasNGPUs(2)) {
        SKIP_TEST("Need at least 2 GPUs");
    }
    EXPECT_TRUE(yali_test::HasNGPUs(2));
}

TEST(GPUsHaveP2PAccess) {
    if (!yali_test::HasNGPUs(2)) {
        SKIP_TEST("Need at least 2 GPUs");
    }
    EXPECT_TRUE(yali_test::HasP2PAccess(0, 1));
    EXPECT_TRUE(yali_test::HasP2PAccess(1, 0));
}

// =============================================================================
// Basic Allocation Tests
// =============================================================================

TEST(CanAllocateGPUMemory) {
    if (!yali_test::HasNGPUs(1)) {
        SKIP_TEST("Need at least 1 GPU");
    }

    void* ptr = nullptr;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&ptr, 1024));
    EXPECT_NE(ptr, nullptr);
    CUDA_CHECK(cudaFree(ptr));
}

TEST(CanAccessPeerMemory) {
    if (!yali_test::HasNGPUs(2) || !yali_test::HasP2PAccess(0, 1)) {
        SKIP_TEST("Need 2 GPUs with P2P access");
    }

    EnableP2P(0, 1);

    // Allocate on GPU 0
    float* ptr0 = nullptr;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&ptr0, 1024 * sizeof(float)));

    // Allocate on GPU 1
    float* ptr1 = nullptr;
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMalloc(&ptr1, 1024 * sizeof(float)));

    // Should be able to memcpy between them
    CUDA_CHECK(cudaMemcpy(ptr1, ptr0, 1024 * sizeof(float), cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(ptr0));
    CUDA_CHECK(cudaFree(ptr1));
}

// =============================================================================
// Tuning Heuristic Tests (these work without kernel linking)
// =============================================================================

TEST(TuningHeuristics_CrossoverBytes) {
    // Test that crossover bytes are reasonable for each dtype
    size_t fp32_crossover = yali::FlashCrossoverBytes(yali::DType::FP32);
    size_t fp16_crossover = yali::FlashCrossoverBytes(yali::DType::FP16);
    size_t bf16_crossover = yali::FlashCrossoverBytes(yali::DType::BF16);

    // Crossover should be positive and reasonable (< 1GB)
    EXPECT_GT(fp32_crossover, 0u);
    EXPECT_LT(fp32_crossover, 1024 * 1024 * 1024u);

    EXPECT_GT(fp16_crossover, 0u);
    EXPECT_LT(fp16_crossover, 1024 * 1024 * 1024u);

    EXPECT_GT(bf16_crossover, 0u);
    EXPECT_LT(bf16_crossover, 1024 * 1024 * 1024u);
}

TEST(TuningHeuristics_LanePresets) {
    // Test that lane presets return valid values
    int ll_lanes_1m = yali::FlashLanePreset(1024 * 1024, yali::DType::FP32);
    int bw_lanes_1m = yali::StreamLanePreset(1024 * 1024, yali::DType::FP32);

    EXPECT_GE(ll_lanes_1m, 1);
    EXPECT_LE(ll_lanes_1m, 128);

    EXPECT_GE(bw_lanes_1m, 1);
    EXPECT_LE(bw_lanes_1m, 128);
}

TEST(TuningHeuristics_SlotBytes) {
    // Test auto slot bytes calculation
    size_t slots_small = yali::AutoSlotBytes(1024);
    size_t slots_medium = yali::AutoSlotBytes(1024 * 1024);
    size_t slots_large = yali::AutoSlotBytes(128 * 1024 * 1024);

    // Should return reasonable positive values
    EXPECT_GT(slots_small, 0u);
    EXPECT_GT(slots_medium, 0u);
    EXPECT_GT(slots_large, 0u);

    // Test clamping
    size_t clamped = yali::ClampSlotBytes(slots_medium, 1024 * 1024);
    EXPECT_GT(clamped, 0u);
}

// =============================================================================
// Low-Latency AllReduce Tests (require kernel library - placeholder for now)
// =============================================================================

// TODO: These tests will work once kernels are extracted to a linkable library
// For now, they are commented out to allow the test binary to compile.

/*
TEST(FlashAllReduce_SmallSize_FP32) {
  if (!yali_test::HasNGPUs(2) || !yali_test::HasP2PAccess(0, 1)) {
    SKIP_TEST("Need 2 GPUs with P2P access");
  }
  // ... kernel launch code ...
}

TEST(FlashAllReduce_MediumSize_FP32) {
  // ...
}

TEST(FlashAllReduce_MultipleLanes_FP32) {
  // ...
}
*/

// =============================================================================
// Main
// =============================================================================

int main() {
    // Check for GPUs first
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
