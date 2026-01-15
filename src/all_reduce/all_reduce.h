// Yali AllReduce public interface
// High-level API for performing AllReduce operations across GPUs
//
// NOTE: We use our own enums (DataType, ReduceOp, KernelMode) instead of
// relying on CUDA/NCCL enums. This keeps the public API clean and portable,
// especially important for C API / Rust FFI bindings (Phase 5).
// Conversion functions to/from NCCL types are provided in internal headers.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace yali {

// Data types supported by Yali AllReduce
// TODO(phase5): Add conversion functions to/from ncclDataType_t
enum class DataType {
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
};

// Reduction operations
enum class ReduceOp {
    Sum = 0,
    // Future: Min, Max, Prod
};

// Kernel mode selection
enum class KernelMode {
    Auto = 0,  // Heuristic-based selection
    Flash,     // Force flash kernel (good for small sizes)
    Stream,    // Force stream kernel (good for large sizes)
};

// AllReduce configuration
struct AllReduceConfig {
    int num_ranks = 2;
    int lanes = -1;          // -1 = auto (heuristic)
    int ctas_per_lane = -1;  // -1 = auto
    KernelMode kernel_mode = KernelMode::Auto;
    int warmup_iters = 1;
    int measure_iters = 5;
};

// AllReduce result / statistics
struct AllReduceResult {
    bool success;
    double latency_us;      // Average latency in microseconds
    double bandwidth_gbps;  // Per-GPU bandwidth in GB/s
    double aggregate_gbps;  // Total aggregate bandwidth
    const char* error_msg;  // nullptr if success
};

// Get element size for a data type
inline size_t ElementSize(DataType dtype) {
    switch (dtype) {
        case DataType::Float16:
        case DataType::BFloat16:
            return 2;
        case DataType::Float32:
        default:
            return 4;
    }
}

// Get string name for data type
inline const char* DataTypeName(DataType dtype) {
    switch (dtype) {
        case DataType::Float16:
            return "fp16";
        case DataType::BFloat16:
            return "bf16";
        case DataType::Float32:
        default:
            return "fp32";
    }
}

// Get string name for kernel mode
inline const char* KernelModeName(KernelMode mode) {
    switch (mode) {
        case KernelMode::Flash:
            return "flash";
        case KernelMode::Stream:
            return "stream";
        case KernelMode::Auto:
        default:
            return "auto";
    }
}

}  // namespace yali
