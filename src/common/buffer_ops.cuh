// Yali buffer operation utilities
// Common GPU buffer operations (seeding, copying, etc.)
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "../kernels/type_ops.cuh"

namespace yali {

// Seed buffer kernel - fills buffer with a constant value
template <typename T>
__global__ void seed_buffer_kernel(T* dst, size_t count, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = ValueOps<T>::FromFloat(value);
    }
}

// Host helper to seed a GPU buffer with a constant value
template <typename T>
inline cudaError_t SeedBuffer(T* dst, size_t count, float value, cudaStream_t stream = 0) {
    if (count == 0)
        return cudaSuccess;
    constexpr int kThreads = 256;
    int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    seed_buffer_kernel<T><<<blocks, kThreads, 0, stream>>>(dst, count, value);
    return cudaGetLastError();
}

// Synchronous version that also synchronizes
template <typename T>
inline cudaError_t SeedBufferSync(T* dst, size_t count, float value) {
    cudaError_t err = SeedBuffer(dst, count, value, 0);
    if (err != cudaSuccess)
        return err;
    return cudaDeviceSynchronize();
}

// Zero buffer
template <typename T>
inline cudaError_t ZeroBuffer(T* dst, size_t count, cudaStream_t stream = 0) {
    if (count == 0)
        return cudaSuccess;
    return cudaMemsetAsync(dst, 0, count * sizeof(T), stream);
}

// Allocate and seed buffer on current device
template <typename T>
inline cudaError_t AllocAndSeed(T** ptr, size_t count, float value) {
    cudaError_t err = cudaMalloc(ptr, count * sizeof(T));
    if (err != cudaSuccess)
        return err;
    return SeedBufferSync(*ptr, count, value);
}

}  // namespace yali
