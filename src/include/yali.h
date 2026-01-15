/*************************************************************************
 * Yali - High-performance AllReduce for 2-GPU NVLink systems
 *
 * This is the public API header. It provides:
 *   - Result codes and data types
 *   - Type conversion utilities
 *   - Documentation of the API design
 *
 * Current Usage:
 *   For now, use the kernel-level API in yali_launch.h with the helpers
 *   in src/all_reduce/kernels.cuh. See examples/basic_all_reduce.cu.
 *
 * Future: The simple AllReduce() API will be implemented to provide
 *   a NCCL-like interface without requiring manual buffer management.
 ************************************************************************/

#ifndef YALI_H_
#define YALI_H_

#include <cuda_runtime.h>

#include <cstddef>

namespace yali {

// Result codes for Yali operations
enum class Result { Success = 0, InvalidArg, CudaError, PeerAccessFailed, NotInitialized };

// Supported data types (matches yali_tuning.h::DType)
enum class Dtype { FP32, FP16, BF16 };

// Get human-readable error string
inline const char* GetErrorString(Result result) {
    switch (result) {
        case Result::Success:
            return "Success";
        case Result::InvalidArg:
            return "Invalid argument";
        case Result::CudaError:
            return "CUDA error";
        case Result::PeerAccessFailed:
            return "Peer access failed";
        case Result::NotInitialized:
            return "Not initialized";
        default:
            return "Unknown error";
    }
}

// Get size in bytes for a dtype
inline size_t DtypeSize(Dtype dtype) {
    switch (dtype) {
        case Dtype::FP32:
            return 4;
        case Dtype::FP16:
            return 2;
        case Dtype::BF16:
            return 2;
        default:
            return 0;
    }
}

// Get dtype name string
inline const char* DtypeName(Dtype dtype) {
    switch (dtype) {
        case Dtype::FP32:
            return "fp32";
        case Dtype::FP16:
            return "fp16";
        case Dtype::BF16:
            return "bf16";
        default:
            return "unknown";
    }
}

}  // namespace yali

#endif  // YALI_H_
