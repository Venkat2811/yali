// Yali validation utilities
// Provides correctness verification for AllReduce results
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "../kernels/type_ops.cuh"

namespace yali {

// Convert typed buffer to float for verification
template <typename T>
__global__ void convert_to_float_kernel(const T* src, float* dst, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = ValueOps<T>::ToFloat(src[idx]);
    }
}

// Host helper to launch conversion kernel
template <typename T>
inline void ConvertBufferToFloat(const T* src, float* dst, size_t count, cudaStream_t stream = 0) {
    if (count == 0)
        return;
    constexpr int kThreads = 256;
    int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    convert_to_float_kernel<T><<<blocks, kThreads, 0, stream>>>(src, dst, count);
}

// Validation result struct
struct ValidationResult {
    bool passed;
    size_t first_mismatch_idx;
    float expected_value;
    float actual_value;
    size_t total_checked;
    size_t mismatches;
};

// Verify AllReduce sum result (host-side)
// For N ranks seeded with values 1,2,...,N the expected sum is N*(N+1)/2
// For 2 ranks: expected = 1 + 2 = 3.0
inline ValidationResult VerifyAllReduceSum(const float* host_data, size_t count, float expected,
                                           float tolerance = 1e-3f) {

    ValidationResult result = {};
    result.passed = true;
    result.expected_value = expected;
    result.total_checked = count;
    result.mismatches = 0;
    result.first_mismatch_idx = SIZE_MAX;

    for (size_t i = 0; i < count; ++i) {
        if (std::fabs(host_data[i] - expected) > tolerance) {
            if (result.first_mismatch_idx == SIZE_MAX) {
                result.first_mismatch_idx = i;
                result.actual_value = host_data[i];
            }
            result.mismatches++;
            result.passed = false;
        }
    }
    return result;
}

// High-level validation for a single rank's receive buffer
// Returns true if validation passed
template <typename T>
inline bool ValidateRankResult(const T* recv_device, size_t elem_count, int rank, int num_ranks,
                               float tolerance = 1e-3f) {

    if (elem_count == 0)
        return true;

    // Expected sum: 1 + 2 + ... + num_ranks = num_ranks * (num_ranks + 1) / 2
    const float expected = static_cast<float>(num_ranks * (num_ranks + 1) / 2);

    // Allocate float buffer for verification
    float* verify_device = nullptr;
    cudaError_t err = cudaMalloc(&verify_device, elem_count * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Validation: cudaMalloc failed for rank %d: %s\n", rank, cudaGetErrorString(err));
        return false;
    }

    // Convert to float
    ConvertBufferToFloat(recv_device, verify_device, elem_count);
    cudaDeviceSynchronize();

    // Copy to host
    std::vector<float> host_data(elem_count);
    err = cudaMemcpy(host_data.data(), verify_device, elem_count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(verify_device);

    if (err != cudaSuccess) {
        fprintf(stderr, "Validation: cudaMemcpy failed for rank %d: %s\n", rank, cudaGetErrorString(err));
        return false;
    }

    // Verify
    ValidationResult result = VerifyAllReduceSum(host_data.data(), elem_count, expected, tolerance);

    if (!result.passed) {
        fprintf(stderr, "Rank %d mismatch at %zu: got %f expected %f (%zu total mismatches)\n", rank,
                result.first_mismatch_idx, result.actual_value, expected, result.mismatches);
    }

    return result.passed;
}

}  // namespace yali
