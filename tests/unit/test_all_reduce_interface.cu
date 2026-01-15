// Unit tests for Yali AllReduce interface types
#include "../../src/all_reduce/all_reduce.h"
#include "test_framework.h"

// =============================================================================
// DataType Tests
// =============================================================================

TEST(DataType_ElementSize_FP32) {
    EXPECT_EQ(yali::ElementSize(yali::DataType::Float32), 4u);
}

TEST(DataType_ElementSize_FP16) {
    EXPECT_EQ(yali::ElementSize(yali::DataType::Float16), 2u);
}

TEST(DataType_ElementSize_BF16) {
    EXPECT_EQ(yali::ElementSize(yali::DataType::BFloat16), 2u);
}

TEST(DataType_Name_FP32) {
    const char* name = yali::DataTypeName(yali::DataType::Float32);
    EXPECT_TRUE(name != nullptr);
    // Check it's "fp32"
    EXPECT_EQ(name[0], 'f');
    EXPECT_EQ(name[1], 'p');
    EXPECT_EQ(name[2], '3');
    EXPECT_EQ(name[3], '2');
}

TEST(DataType_Name_FP16) {
    const char* name = yali::DataTypeName(yali::DataType::Float16);
    EXPECT_TRUE(name != nullptr);
    EXPECT_EQ(name[0], 'f');
    EXPECT_EQ(name[1], 'p');
    EXPECT_EQ(name[2], '1');
    EXPECT_EQ(name[3], '6');
}

TEST(DataType_Name_BF16) {
    const char* name = yali::DataTypeName(yali::DataType::BFloat16);
    EXPECT_TRUE(name != nullptr);
    EXPECT_EQ(name[0], 'b');
    EXPECT_EQ(name[1], 'f');
    EXPECT_EQ(name[2], '1');
    EXPECT_EQ(name[3], '6');
}

// =============================================================================
// KernelMode Tests
// =============================================================================

TEST(KernelMode_Name_Auto) {
    const char* name = yali::KernelModeName(yali::KernelMode::Auto);
    EXPECT_TRUE(name != nullptr);
    EXPECT_EQ(name[0], 'a');
    EXPECT_EQ(name[1], 'u');
    EXPECT_EQ(name[2], 't');
    EXPECT_EQ(name[3], 'o');
}

TEST(KernelMode_Name_Flash) {
    const char* name = yali::KernelModeName(yali::KernelMode::Flash);
    EXPECT_TRUE(name != nullptr);
    EXPECT_EQ(name[0], 'f');
    EXPECT_EQ(name[1], 'l');
    EXPECT_EQ(name[2], 'a');
}

TEST(KernelMode_Name_Stream) {
    const char* name = yali::KernelModeName(yali::KernelMode::Stream);
    EXPECT_TRUE(name != nullptr);
    EXPECT_EQ(name[0], 's');
    EXPECT_EQ(name[1], 't');
    EXPECT_EQ(name[2], 'r');
}

// =============================================================================
// AllReduceConfig Tests
// =============================================================================

TEST(AllReduceConfig_Defaults) {
    yali::AllReduceConfig config;
    EXPECT_EQ(config.num_ranks, 2);
    EXPECT_EQ(config.lanes, -1);          // Auto
    EXPECT_EQ(config.ctas_per_lane, -1);  // Auto
    EXPECT_EQ(static_cast<int>(config.kernel_mode), static_cast<int>(yali::KernelMode::Auto));
    EXPECT_EQ(config.warmup_iters, 1);
    EXPECT_EQ(config.measure_iters, 5);
}

// =============================================================================
// AllReduceResult Tests
// =============================================================================

TEST(AllReduceResult_SuccessInit) {
    yali::AllReduceResult result = {};
    result.success = true;
    result.latency_us = 100.0;
    result.bandwidth_gbps = 50.0;
    result.aggregate_gbps = 100.0;
    result.error_msg = nullptr;

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.error_msg, nullptr);
}

TEST(AllReduceResult_FailureInit) {
    yali::AllReduceResult result = {};
    result.success = false;
    result.error_msg = "Test error";

    EXPECT_FALSE(result.success);
    EXPECT_NE(result.error_msg, nullptr);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("Testing AllReduce interface types\n");
    return RUN_ALL_TESTS();
}
