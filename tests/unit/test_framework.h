// Minimal CUDA test framework - no external dependencies
// Usage:
//   TEST(TestName) { ... assertions ... }
//   EXPECT_TRUE(cond);
//   EXPECT_EQ(a, b);
//   EXPECT_NEAR(a, b, tolerance);
//   CUDA_CHECK(err);

#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace yali_test {

// Test result tracking
struct TestResult {
    std::string name;
    bool passed;
    std::string error_msg;
};

static std::vector<TestResult> g_test_results;
static std::string g_current_test;
static bool g_current_passed;
static std::string g_current_error;

// CUDA error checking macro
#define CUDA_CHECK(err)                                                                                                \
    do {                                                                                                               \
        cudaError_t _err = (err);                                                                                      \
        if (_err != cudaSuccess) {                                                                                     \
            yali_test::g_current_passed = false;                                                                       \
            char buf[512];                                                                                             \
            snprintf(buf, sizeof(buf), "CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(_err));       \
            yali_test::g_current_error = buf;                                                                          \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

// Assertion macros
#define EXPECT_TRUE(cond)                                                                                              \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            yali_test::g_current_passed = false;                                                                       \
            char buf[512];                                                                                             \
            snprintf(buf, sizeof(buf), "EXPECT_TRUE failed at %s:%d: %s", __FILE__, __LINE__, #cond);                  \
            yali_test::g_current_error = buf;                                                                          \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define EXPECT_FALSE(cond) EXPECT_TRUE(!(cond))

#define EXPECT_EQ(a, b)                                                                                                \
    do {                                                                                                               \
        auto _a = (a);                                                                                                 \
        auto _b = (b);                                                                                                 \
        if (_a != _b) {                                                                                                \
            yali_test::g_current_passed = false;                                                                       \
            char buf[512];                                                                                             \
            snprintf(buf, sizeof(buf), "EXPECT_EQ failed at %s:%d: %s != %s", __FILE__, __LINE__, #a, #b);             \
            yali_test::g_current_error = buf;                                                                          \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define EXPECT_NE(a, b)                                                                                                \
    do {                                                                                                               \
        auto _a = (a);                                                                                                 \
        auto _b = (b);                                                                                                 \
        if (_a == _b) {                                                                                                \
            yali_test::g_current_passed = false;                                                                       \
            char buf[512];                                                                                             \
            snprintf(buf, sizeof(buf), "EXPECT_NE failed at %s:%d: %s == %s", __FILE__, __LINE__, #a, #b);             \
            yali_test::g_current_error = buf;                                                                          \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define EXPECT_LT(a, b)                                                                                                \
    do {                                                                                                               \
        auto _a = (a);                                                                                                 \
        auto _b = (b);                                                                                                 \
        if (!(_a < _b)) {                                                                                              \
            yali_test::g_current_passed = false;                                                                       \
            char buf[512];                                                                                             \
            snprintf(buf, sizeof(buf), "EXPECT_LT failed at %s:%d: %s >= %s", __FILE__, __LINE__, #a, #b);             \
            yali_test::g_current_error = buf;                                                                          \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define EXPECT_LE(a, b)                                                                                                \
    do {                                                                                                               \
        auto _a = (a);                                                                                                 \
        auto _b = (b);                                                                                                 \
        if (!(_a <= _b)) {                                                                                             \
            yali_test::g_current_passed = false;                                                                       \
            char buf[512];                                                                                             \
            snprintf(buf, sizeof(buf), "EXPECT_LE failed at %s:%d: %s > %s", __FILE__, __LINE__, #a, #b);              \
            yali_test::g_current_error = buf;                                                                          \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define EXPECT_GT(a, b)                                                                                                \
    do {                                                                                                               \
        auto _a = (a);                                                                                                 \
        auto _b = (b);                                                                                                 \
        if (!(_a > _b)) {                                                                                              \
            yali_test::g_current_passed = false;                                                                       \
            char buf[512];                                                                                             \
            snprintf(buf, sizeof(buf), "EXPECT_GT failed at %s:%d: %s <= %s", __FILE__, __LINE__, #a, #b);             \
            yali_test::g_current_error = buf;                                                                          \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define EXPECT_GE(a, b)                                                                                                \
    do {                                                                                                               \
        auto _a = (a);                                                                                                 \
        auto _b = (b);                                                                                                 \
        if (!(_a >= _b)) {                                                                                             \
            yali_test::g_current_passed = false;                                                                       \
            char buf[512];                                                                                             \
            snprintf(buf, sizeof(buf), "EXPECT_GE failed at %s:%d: %s < %s", __FILE__, __LINE__, #a, #b);              \
            yali_test::g_current_error = buf;                                                                          \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define EXPECT_NEAR(a, b, tolerance)                                                                                   \
    do {                                                                                                               \
        double _a = static_cast<double>(a);                                                                            \
        double _b = static_cast<double>(b);                                                                            \
        double _tol = static_cast<double>(tolerance);                                                                  \
        if (std::fabs(_a - _b) > _tol) {                                                                               \
            yali_test::g_current_passed = false;                                                                       \
            char buf[512];                                                                                             \
            snprintf(buf, sizeof(buf), "EXPECT_NEAR failed at %s:%d: |%s - %s| = %g > %g", __FILE__, __LINE__, #a, #b, \
                     std::fabs(_a - _b), _tol);                                                                        \
            yali_test::g_current_error = buf;                                                                          \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

// Skip test (for conditional skipping)
#define SKIP_TEST(reason)                                                                                              \
    do {                                                                                                               \
        printf("  SKIPPED: %s\n", reason);                                                                             \
        yali_test::g_current_passed = true;                                                                            \
        return;                                                                                                        \
    } while (0)

// Test registration
struct TestRegistry {
    std::string name;
    std::function<void()> func;
};

static std::vector<TestRegistry> g_tests;

struct TestRegistrar {
    TestRegistrar(const char* name, std::function<void()> func) { g_tests.push_back({name, func}); }
};

#define TEST(name)                                                                                                     \
    static void test_##name();                                                                                         \
    static yali_test::TestRegistrar reg_##name(#name, test_##name);                                                    \
    static void test_##name()

// Run all tests
inline int RunAllTests() {
    int passed = 0;
    int failed = 0;
    int total = static_cast<int>(g_tests.size());

    printf("Running %d tests...\n", total);
    printf("========================================\n");

    for (const auto& test : g_tests) {
        g_current_test = test.name;
        g_current_passed = true;
        g_current_error.clear();

        printf("[ RUN      ] %s\n", test.name.c_str());

        // Run the test
        test.func();

        // Record result
        g_test_results.push_back({test.name, g_current_passed, g_current_error});

        if (g_current_passed) {
            printf("[       OK ] %s\n", test.name.c_str());
            passed++;
        } else {
            printf("[  FAILED  ] %s\n", test.name.c_str());
            if (!g_current_error.empty()) {
                printf("             %s\n", g_current_error.c_str());
            }
            failed++;
        }
    }

    printf("========================================\n");
    printf("[==========] %d tests ran.\n", total);
    printf("[  PASSED  ] %d tests.\n", passed);
    if (failed > 0) {
        printf("[  FAILED  ] %d tests.\n", failed);
    }

    return failed == 0 ? 0 : 1;
}

// Utility: Check if at least N GPUs are available
inline bool HasNGPUs(int n) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count >= n;
}

// Utility: Check if P2P access is available between two GPUs
inline bool HasP2PAccess(int dev0, int dev1) {
    int canAccess = 0;
    cudaError_t err = cudaDeviceCanAccessPeer(&canAccess, dev0, dev1);
    return err == cudaSuccess && canAccess;
}

}  // namespace yali_test

// Main function macro for test executables
#define RUN_ALL_TESTS() yali_test::RunAllTests()
