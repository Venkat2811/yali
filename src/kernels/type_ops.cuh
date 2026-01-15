/*************************************************************************
 * Common datatype helpers for Yali kernels.
 *
 * Vectorized operations for FP16/BF16 to achieve full NVLink bandwidth.
 ************************************************************************/

#ifndef YALI_KERNELS_TYPE_OPS_CUH_
#define YALI_KERNELS_TYPE_OPS_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdint.h>

namespace yali {

// ============================================================================
// Scalar value operations (used for validation, not performance-critical)
// ============================================================================

template <typename T>
struct ValueOps;

template <>
struct ValueOps<float> {
    using Scalar = float;
    static __host__ __device__ inline Scalar FromFloat(float v) { return v; }
    static __host__ __device__ inline float ToFloat(Scalar v) { return v; }
    static __host__ __device__ inline Scalar Add(Scalar a, Scalar b) { return a + b; }
};

template <>
struct ValueOps<__half> {
    using Scalar = __half;
    static __host__ __device__ inline Scalar FromFloat(float v) { return __float2half(v); }
    static __host__ __device__ inline float ToFloat(Scalar v) { return __half2float(v); }
    static __device__ inline Scalar Add(Scalar a, Scalar b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
        return __hadd(a, b);
#else
        return __float2half(__half2float(a) + __half2float(b));
#endif
    }
};

template <>
struct ValueOps<__nv_bfloat16> {
    using Scalar = __nv_bfloat16;
    static __host__ __device__ inline Scalar FromFloat(float v) { return __float2bfloat16(v); }
    static __host__ __device__ inline float ToFloat(Scalar v) { return __bfloat162float(v); }
    static __device__ inline Scalar Add(Scalar a, Scalar b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        return __hadd(a, b);
#else
        return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
#endif
    }
};

// ============================================================================
// Vectorized reduction helpers for 16-bit types
// These use uint4 (16-byte) loads/stores with in-register reduction
// ============================================================================

// Helper to add two half2 vectors
__device__ __forceinline__ __half2 add_half2(__half2 a, __half2 b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    return __hadd2(a, b);
#else
    float2 fa = __half22float2(a);
    float2 fb = __half22float2(b);
    return __float22half2_rn(make_float2(fa.x + fb.x, fa.y + fb.y));
#endif
}

// Helper to add two bfloat162 vectors
__device__ __forceinline__ __nv_bfloat162 add_bfloat162(__nv_bfloat162 a, __nv_bfloat162 b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return __hadd2(a, b);
#else
    float fa_lo = __bfloat162float(__low2bfloat16(a));
    float fa_hi = __bfloat162float(__high2bfloat16(a));
    float fb_lo = __bfloat162float(__low2bfloat16(b));
    float fb_hi = __bfloat162float(__high2bfloat16(b));
    return __floats2bfloat162_rn(fa_lo + fb_lo, fa_hi + fb_hi);
#endif
}

// ============================================================================
// Helper: Coalesced copy with proper tail handling
// Avoids byte-by-byte copy which destroys memory bandwidth
// ============================================================================

__device__ __forceinline__ void coalesced_copy_16(char* dst, const void* src, int32_t bytes) {
    constexpr int32_t kVectorBytes = 16;

    // Main loop: 16-byte vectorized copies (uint4)
    const int32_t vecCount = bytes / kVectorBytes;
    auto dstVec = reinterpret_cast<uint4*>(dst);
    auto srcVec = reinterpret_cast<const uint4*>(src);
    for (int32_t i = threadIdx.x; i < vecCount; i += blockDim.x) {
        dstVec[i] = srcVec[i];
    }

    // Tail handling: Use progressively smaller coalesced accesses
    // Only thread 0 handles tail to avoid race conditions
    int32_t offset = vecCount * kVectorBytes;

    if (threadIdx.x == 0) {
        // 8-byte tail (uint2)
        if (offset + 8 <= bytes) {
            *reinterpret_cast<uint2*>(dst + offset) =
                *reinterpret_cast<const uint2*>(reinterpret_cast<const char*>(src) + offset);
            offset += 8;
        }

        // 4-byte tail (uint)
        if (offset + 4 <= bytes) {
            *reinterpret_cast<uint32_t*>(dst + offset) =
                *reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(src) + offset);
            offset += 4;
        }

        // 2-byte tail (ushort)
        if (offset + 2 <= bytes) {
            *reinterpret_cast<uint16_t*>(dst + offset) =
                *reinterpret_cast<const uint16_t*>(reinterpret_cast<const char*>(src) + offset);
            offset += 2;
        }

        // 1-byte tail (last resort)
        if (offset < bytes) {
            dst[offset] = reinterpret_cast<const char*>(src)[offset];
        }
    }
}

// ============================================================================
// SlotOps: Vectorized copy and reduce operations
// ============================================================================

template <typename T>
struct SlotOps {
    static constexpr int kVectorBytes = 16;
    static_assert(kVectorBytes % sizeof(T) == 0, "Vector bytes must align with element size");
    static constexpr int kElemsPerVec = kVectorBytes / sizeof(T);

    __device__ static inline void copy(char* dst, const void* src, int32_t bytes) {
        coalesced_copy_16(dst, src, bytes);
    }

    // Default reduce: scalar operations (used for FP32)
    __device__ static inline void reduce(void* dst, const char* src, int32_t bytes) {
        T* dstElems = reinterpret_cast<T*>(dst);
        const T* srcElems = reinterpret_cast<const T*>(src);
        const int32_t totalElems = bytes / static_cast<int32_t>(sizeof(T));
        for (int32_t i = threadIdx.x; i < totalElems; i += blockDim.x) {
            dstElems[i] = ValueOps<T>::Add(dstElems[i], srcElems[i]);
        }
        // Tail bytes are ignored for reduce - should be element-aligned
    }
};

// ============================================================================
// Specialized SlotOps for FP16 (__half) - uses vectorized uint4 loads with
// in-register half2 reduction for full memory bandwidth
// ============================================================================

template <>
struct SlotOps<__half> {
    static constexpr int kVectorBytes = 16;
    static constexpr int kElemsPerVec = kVectorBytes / sizeof(__half);  // 8 half per uint4

    __device__ static inline void copy(char* dst, const void* src, int32_t bytes) {
        coalesced_copy_16(dst, src, bytes);
    }

    // Vectorized reduce: load uint4 (16 bytes = 8 half), reduce as 4x half2, store uint4
    __device__ static inline void reduce(void* dst, const char* src, int32_t bytes) {
        const int32_t vecCount = bytes / kVectorBytes;
        uint4* dstVec = reinterpret_cast<uint4*>(dst);
        const uint4* srcVec = reinterpret_cast<const uint4*>(src);

        for (int32_t i = threadIdx.x; i < vecCount; i += blockDim.x) {
            // Load 16 bytes (8 half elements) from both src and dst
            uint4 d = dstVec[i];
            uint4 s = srcVec[i];

            // Reinterpret as half2 arrays (4 x half2 = 8 half = 16 bytes)
            __half2* d_h2 = reinterpret_cast<__half2*>(&d);
            const __half2* s_h2 = reinterpret_cast<const __half2*>(&s);

            // Reduce each half2 pair
            d_h2[0] = add_half2(d_h2[0], s_h2[0]);
            d_h2[1] = add_half2(d_h2[1], s_h2[1]);
            d_h2[2] = add_half2(d_h2[2], s_h2[2]);
            d_h2[3] = add_half2(d_h2[3], s_h2[3]);

            // Store result
            dstVec[i] = d;
        }

        // Handle tail elements (not aligned to 16 bytes) - use half2 where possible
        const int32_t vecBytes = vecCount * kVectorBytes;
        if (vecBytes < bytes) {
            __half* dstTail = reinterpret_cast<__half*>(reinterpret_cast<char*>(dst) + vecBytes);
            const __half* srcTail = reinterpret_cast<const __half*>(src + vecBytes);
            const int32_t tailElems = (bytes - vecBytes) / sizeof(__half);

            // Handle pairs first (half2) for better throughput
            const int32_t pairCount = tailElems / 2;
            __half2* dstPairs = reinterpret_cast<__half2*>(dstTail);
            const __half2* srcPairs = reinterpret_cast<const __half2*>(srcTail);
            for (int32_t i = threadIdx.x; i < pairCount; i += blockDim.x) {
                dstPairs[i] = add_half2(dstPairs[i], srcPairs[i]);
            }

            // Handle odd element (if any)
            if (tailElems % 2 == 1 && threadIdx.x == 0) {
                const int32_t lastIdx = tailElems - 1;
                dstTail[lastIdx] = ValueOps<__half>::Add(dstTail[lastIdx], srcTail[lastIdx]);
            }
        }
    }
};

// ============================================================================
// Specialized SlotOps for BF16 (__nv_bfloat16) - uses vectorized uint4 loads
// with in-register bfloat162 reduction for full memory bandwidth
// ============================================================================

template <>
struct SlotOps<__nv_bfloat16> {
    static constexpr int kVectorBytes = 16;
    static constexpr int kElemsPerVec = kVectorBytes / sizeof(__nv_bfloat16);  // 8 bf16 per uint4

    __device__ static inline void copy(char* dst, const void* src, int32_t bytes) {
        coalesced_copy_16(dst, src, bytes);
    }

    // Vectorized reduce: load uint4 (16 bytes = 8 bf16), reduce as 4x bfloat162, store uint4
    __device__ static inline void reduce(void* dst, const char* src, int32_t bytes) {
        const int32_t vecCount = bytes / kVectorBytes;
        uint4* dstVec = reinterpret_cast<uint4*>(dst);
        const uint4* srcVec = reinterpret_cast<const uint4*>(src);

        for (int32_t i = threadIdx.x; i < vecCount; i += blockDim.x) {
            // Load 16 bytes (8 bf16 elements) from both src and dst
            uint4 d = dstVec[i];
            uint4 s = srcVec[i];

            // Reinterpret as bfloat162 arrays (4 x bfloat162 = 8 bf16 = 16 bytes)
            __nv_bfloat162* d_bf2 = reinterpret_cast<__nv_bfloat162*>(&d);
            const __nv_bfloat162* s_bf2 = reinterpret_cast<const __nv_bfloat162*>(&s);

            // Reduce each bfloat162 pair
            d_bf2[0] = add_bfloat162(d_bf2[0], s_bf2[0]);
            d_bf2[1] = add_bfloat162(d_bf2[1], s_bf2[1]);
            d_bf2[2] = add_bfloat162(d_bf2[2], s_bf2[2]);
            d_bf2[3] = add_bfloat162(d_bf2[3], s_bf2[3]);

            // Store result
            dstVec[i] = d;
        }

        // Handle tail elements (not aligned to 16 bytes) - use bfloat162 where possible
        const int32_t vecBytes = vecCount * kVectorBytes;
        if (vecBytes < bytes) {
            __nv_bfloat16* dstTail = reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<char*>(dst) + vecBytes);
            const __nv_bfloat16* srcTail = reinterpret_cast<const __nv_bfloat16*>(src + vecBytes);
            const int32_t tailElems = (bytes - vecBytes) / sizeof(__nv_bfloat16);

            // Handle pairs first (bfloat162) for better throughput
            const int32_t pairCount = tailElems / 2;
            __nv_bfloat162* dstPairs = reinterpret_cast<__nv_bfloat162*>(dstTail);
            const __nv_bfloat162* srcPairs = reinterpret_cast<const __nv_bfloat162*>(srcTail);
            for (int32_t i = threadIdx.x; i < pairCount; i += blockDim.x) {
                dstPairs[i] = add_bfloat162(dstPairs[i], srcPairs[i]);
            }

            // Handle odd element (if any)
            if (tailElems % 2 == 1 && threadIdx.x == 0) {
                const int32_t lastIdx = tailElems - 1;
                dstTail[lastIdx] = ValueOps<__nv_bfloat16>::Add(dstTail[lastIdx], srcTail[lastIdx]);
            }
        }
    }
};

}  // namespace yali

#endif  // YALI_KERNELS_TYPE_OPS_CUH_
