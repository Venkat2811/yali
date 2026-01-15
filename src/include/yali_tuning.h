/*************************************************************************
 * Yali tuning heuristics for the standalone harness.
 *
 * Updated 2025-12-02 based on NV2 (2-NVLink A100) sweep results.
 * Crossover points (LL vs BW):
 *   - FP32: 64M (BW wins at 64M+)
 *   - FP16: 64M (BW wins at 64M+)
 *   - BF16: 64M (LL marginally wins at 64M, BW wins at 128M+)
 ************************************************************************/

#ifndef YALI_TUNING_H_
#define YALI_TUNING_H_

#include <algorithm>
#include <cmath>
#include <stddef.h>
#include <stdint.h>

namespace yali {

// Dtype identifiers for heuristic selection
enum class DType { FP32 = 0, FP16 = 1, BF16 = 2 };

// Crossover threshold: use Low-Latency below this, Bandwidth above.
// Based on NV2 sweep 2025-12-02.
inline size_t FlashCrossoverBytes(DType dtype) {
    switch (dtype) {
        case DType::FP32:
            return 64ull << 20;  // 64M
        case DType::FP16:
            return 64ull << 20;  // 64M
        case DType::BF16:
            return 64ull << 20;  // 64M (LL marginally better, but close)
        default:
            return 64ull << 20;
    }
}

// Optimal lane count for Bandwidth kernel based on sweep data.
// Larger messages benefit from more lanes (up to 128).
inline int StreamLanePreset(size_t bytes, DType /*dtype*/) {
    // From NV2 sweep: 128 lanes optimal for sizes >= 128M
    if (bytes >= (128ull << 20))
        return 128;
    if (bytes >= (64ull << 20))
        return 64;
    if (bytes >= (16ull << 20))
        return 32;
    return 16;
}

// Optimal lane count for Low-Latency kernel based on sweep data.
// Tuned per dtype from NV2 sweep 2025-12-02.
inline int FlashLanePreset(size_t bytes, DType dtype) {
    // FP32 optimal lanes from sweep:
    // 2K-256K: 16-32, 1M: 16, 4M: 64, 16M: 16, 64M: 64
    if (dtype == DType::FP32) {
        if (bytes <= (256ull << 10))
            return 16;  // <=256K
        if (bytes <= (1ull << 20))
            return 16;  // 1M
        if (bytes <= (4ull << 20))
            return 64;  // 4M
        if (bytes <= (16ull << 20))
            return 16;  // 16M
        if (bytes <= (64ull << 20))
            return 64;  // 64M
        return 128;     // >64M (BW mode, but fallback)
    }
    // FP16 optimal lanes from sweep:
    // 2K-256K: 16, 1M: 32, 4M: 32, 16M: 16, 64M: 32
    if (dtype == DType::FP16) {
        if (bytes <= (256ull << 10))
            return 16;  // <=256K
        if (bytes <= (4ull << 20))
            return 32;  // 1M-4M
        if (bytes <= (16ull << 20))
            return 16;  // 16M
        if (bytes <= (64ull << 20))
            return 32;  // 64M
        return 128;
    }
    // BF16 optimal lanes from sweep:
    // Similar to FP16: 2K-256K: 16, 1M: 32, 4M: 32, 16M: 16, 64M: 32
    if (dtype == DType::BF16) {
        if (bytes <= (256ull << 10))
            return 16;  // <=256K
        if (bytes <= (4ull << 20))
            return 32;  // 1M-4M
        if (bytes <= (16ull << 20))
            return 16;  // 16M
        if (bytes <= (64ull << 20))
            return 32;  // 64M
        return 128;
    }
    // Fallback
    return 32;
}

inline size_t AutoSlotBytes(size_t bytes) {
    if (bytes <= (8ull << 20)) {
        return 64ull << 10;
    } else if (bytes <= (32ull << 20)) {
        return 128ull << 10;
    } else if (bytes <= (64ull << 20)) {
        return 256ull << 10;
    }
    return 1ull << 20;
}

inline size_t ClampSlotBytes(size_t slotBytes, size_t maxBytes) {
    size_t clamped = std::max<size_t>(256, std::min(slotBytes, maxBytes));
    if (clamped > static_cast<size_t>(INT32_MAX))
        clamped = static_cast<size_t>(INT32_MAX);
    return clamped;
}

inline int AutoCtasPerLane(bool useFlash, int lanes, size_t maxLaneElemCount, size_t tileElems) {
    if (!useFlash)
        return 1;
    if (maxLaneElemCount == 0 || lanes <= 0)
        return 1;
    if (tileElems == 0)
        tileElems = 1;
    size_t needed = (maxLaneElemCount + tileElems - 1) / tileElems;
    size_t minNeeded = 4;
    if (needed < minNeeded)
        needed = minNeeded;
    if (needed > 32)
        needed = 32;
    return static_cast<int>(needed);
}

}  // namespace yali

#endif  // YALI_TUNING_H_
