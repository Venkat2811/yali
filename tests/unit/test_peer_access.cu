// Unit tests for Yali peer access utilities
#include <cuda_runtime.h>

#include "../../src/common/peer_access.cuh"
#include "test_framework.h"

// =============================================================================
// P2P Capability Tests
// =============================================================================

TEST(CheckP2PAccess_TwoGPUs) {
    if (!yali_test::HasNGPUs(2)) {
        SKIP_TEST("Need at least 2 GPUs");
    }

    auto cap = yali::CheckP2PAccess(0, 1);
    EXPECT_EQ(cap.device0, 0);
    EXPECT_EQ(cap.device1, 1);
    // On systems with NVLink, both should be true
    // We don't assert the values since it depends on hardware
    printf("  P2P 0->1: %s, 1->0: %s, bidirectional: %s\n", cap.can_access_0_to_1 ? "yes" : "no",
           cap.can_access_1_to_0 ? "yes" : "no", cap.bidirectional ? "yes" : "no");
}

TEST(CheckP2PAccess_Consistency) {
    if (!yali_test::HasNGPUs(2)) {
        SKIP_TEST("Need at least 2 GPUs");
    }

    // Check that bidirectional is consistent with the two directions
    auto cap = yali::CheckP2PAccess(0, 1);
    bool expected_bidir = cap.can_access_0_to_1 && cap.can_access_1_to_0;
    EXPECT_EQ(cap.bidirectional, expected_bidir);
}

// =============================================================================
// Enable P2P Tests
// =============================================================================

TEST(EnableP2PAccess_Success) {
    if (!yali_test::HasNGPUs(2) || !yali_test::HasP2PAccess(0, 1)) {
        SKIP_TEST("Need 2 GPUs with P2P access");
    }

    cudaError_t err = yali::EnableP2PAccess(0, 1);
    EXPECT_EQ(err, cudaSuccess);

    // Calling again should also succeed (already enabled case)
    err = yali::EnableP2PAccess(0, 1);
    EXPECT_EQ(err, cudaSuccess);
}

TEST(EnableBidirectionalP2P_Success) {
    if (!yali_test::HasNGPUs(2) || !yali_test::HasP2PAccess(0, 1)) {
        SKIP_TEST("Need 2 GPUs with P2P access");
    }

    bool result = yali::EnableBidirectionalP2P(0, 1);
    EXPECT_TRUE(result);
}

TEST(EnableBidirectionalP2P_Idempotent) {
    if (!yali_test::HasNGPUs(2) || !yali_test::HasP2PAccess(0, 1)) {
        SKIP_TEST("Need 2 GPUs with P2P access");
    }

    // Enable twice - should still return true
    bool result1 = yali::EnableBidirectionalP2P(0, 1);
    bool result2 = yali::EnableBidirectionalP2P(0, 1);
    EXPECT_TRUE(result1);
    EXPECT_TRUE(result2);
}

// =============================================================================
// Topology Tests
// =============================================================================

TEST(QueryTopology_DeviceCount) {
    auto info = yali::QueryTopology();
    int expected_count = 0;
    cudaGetDeviceCount(&expected_count);
    EXPECT_EQ(info.device_count, expected_count);
}

TEST(QueryTopology_PairCount) {
    if (!yali_test::HasNGPUs(2)) {
        SKIP_TEST("Need at least 2 GPUs");
    }

    auto info = yali::QueryTopology();
    // For N devices, we have N*(N-1)/2 pairs
    int expected_pairs = info.device_count * (info.device_count - 1) / 2;
    EXPECT_EQ(static_cast<int>(info.p2p_capabilities.size()), expected_pairs);
}

TEST(AllP2PEnabled_TwoGPUsWithNVLink) {
    if (!yali_test::HasNGPUs(2) || !yali_test::HasP2PAccess(0, 1)) {
        SKIP_TEST("Need 2 GPUs with P2P access");
    }

    auto info = yali::QueryTopology();
    // If we have P2P access, AllP2PEnabled should return true
    EXPECT_TRUE(info.AllP2PEnabled());
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
