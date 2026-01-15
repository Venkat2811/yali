/*************************************************************************
 * Hardware Info Detection Utilities
 *
 * Provides runtime detection of NVLink configuration and theoretical
 * bandwidth limits for Speed-of-Light (SoL) calculations.
 *
 * SoL Calculation Reference:
 * ---------------------------
 * Based on nvbandwidth D2D measurements and NCCL-tests methodology.
 *
 * Key concepts:
 * - Unidirectional NVLink BW: What one GPU can read from another (e.g., 100 GB/s for 4xNVLink)
 * - Bidirectional NVLink BW: Both GPUs reading/writing simultaneously (2x unidirectional)
 * - Algorithm BW (algbw): data_size / time
 * - Bus BW (busbw): For AllReduce ring, busbw = algbw * (n-1)/n (for 2 GPUs: busbw = algbw)
 *
 * Yali achieves >100% of unidirectional SoL because it uses bidirectional NVLink
 * (both GPUs reading from each other simultaneously). This is expected and correct.
 *
 * Example (A100 4xNVLink):
 * - Unidirectional peak: 4 x 25 GB/s = 100 GB/s
 * - Bidirectional peak: 2 x 100 GB/s = 200 GB/s (both directions)
 * - Yali @ 2G: ~165 GB/s algbw = 165% of unidirectional SoL (82.5% of bidirectional)
 ************************************************************************/

#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace yali {

/**
 * NVLink configuration detected at runtime
 */
struct NVLinkInfo {
    int linkCount;             // Number of NVLinks between GPUs (from nvidia-smi nvlink -s)
    double bwPerLinkGBs;       // Bandwidth per link in GB/s (25.0 for NVLink 3.0/4.0)
    double unidirectionalGBs;  // Unidirectional bandwidth (what nvbandwidth D2D measures)
    double bidirectionalGBs;   // Bidirectional bandwidth (both GPUs reading simultaneously)
    bool valid;                // Whether detection succeeded
};

/**
 * Detect NVLink configuration by parsing nvidia-smi nvlink -s output
 *
 * Example output:
 *   GPU 0: NVIDIA A100-SXM4-80GB (UUID: ...)
 *        Link 0: 25 GB/s
 *        Link 1: 25 GB/s
 *        ...
 *
 * Returns NVLinkInfo with detected values, or fallback defaults on failure.
 */
inline NVLinkInfo DetectNVLinkConfig() {
    NVLinkInfo info = {0, 25.0, 0.0, 0.0, false};

    // Run nvidia-smi nvlink -s and parse output
    FILE* pipe = popen("nvidia-smi nvlink -s 2>/dev/null", "r");
    if (!pipe) {
        // Fallback: use cudaDeviceProp to infer
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            if (prop.major >= 9) {
                info.linkCount = 18;  // H100 NVSwitch
            } else if (prop.major == 8) {
                info.linkCount = 4;  // A100 SXM4 (conservative default)
            } else {
                info.linkCount = 6;  // V100
            }
            info.unidirectionalGBs = info.linkCount * info.bwPerLinkGBs;
            info.bidirectionalGBs = info.unidirectionalGBs * 2.0;
            info.valid = true;
        }
        return info;
    }

    char line[256];
    int currentGpu = -1;
    int gpu0LinkCount = 0;
    double gpu0BwSum = 0.0;

    while (fgets(line, sizeof(line), pipe)) {
        // Parse "GPU 0: NVIDIA A100..."
        if (strncmp(line, "GPU ", 4) == 0) {
            int gpuIdx = 0;
            if (sscanf(line, "GPU %d:", &gpuIdx) == 1) {
                currentGpu = gpuIdx;
            }
        }
        // Parse "     Link N: XX GB/s"
        else if (currentGpu == 0 && strstr(line, "Link ") && strstr(line, "GB/s")) {
            int linkNum = 0;
            double bw = 0.0;
            // Find the link number and bandwidth
            char* linkPtr = strstr(line, "Link ");
            if (linkPtr) {
                if (sscanf(linkPtr, "Link %d: %lf GB/s", &linkNum, &bw) == 2) {
                    gpu0LinkCount++;
                    gpu0BwSum += bw;
                }
            }
        }
    }
    pclose(pipe);

    if (gpu0LinkCount > 0) {
        info.linkCount = gpu0LinkCount;
        info.bwPerLinkGBs = gpu0BwSum / gpu0LinkCount;  // Average (should all be same)
        info.unidirectionalGBs = gpu0BwSum;
        info.bidirectionalGBs = gpu0BwSum * 2.0;  // Both directions simultaneously
        info.valid = true;
    } else {
        // Fallback to architecture-based detection
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            if (prop.major >= 9) {
                info.linkCount = 18;
            } else if (prop.major == 8) {
                info.linkCount = 4;
            } else {
                info.linkCount = 6;
            }
            info.unidirectionalGBs = info.linkCount * info.bwPerLinkGBs;
            info.bidirectionalGBs = info.unidirectionalGBs * 2.0;
            info.valid = true;
        }
    }

    return info;
}

/**
 * Get GPU name from cudaDeviceProp
 */
inline const char* GetGpuName(int device = 0) {
    static char name[256] = {0};
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
        strncpy(name, prop.name, sizeof(name) - 1);
        return name;
    }
    return "Unknown GPU";
}

/**
 * Calculate Speed-of-Light metrics for AllReduce
 *
 * Bandwidth calculation (NCCL convention):
 * - algbw = data_size / time (algorithm bandwidth)
 * - busbw = algbw * (n-1)/n for ring (for 2 GPUs: busbw = algbw)
 *
 * SoL calculation:
 * - SoL% is relative to UNIDIRECTIONAL NVLink bandwidth (nvbandwidth D2D reference)
 * - Values >100% are expected and correct for Yali because it uses bidirectional NVLink
 * - This matches the methodology in benchmark comparison files
 *
 * For 2-GPU AllReduce:
 * - Each GPU reads bytes from peer via NVLink (unidirectional: 100 GB/s for 4xNVLink)
 * - Yali uses bidirectional NVLink (both GPUs reading simultaneously)
 * - Theoretical bidirectional peak = 2 x unidirectional = 200 GB/s
 * - Yali typically achieves 150-170% of unidirectional SoL at large sizes
 */
struct SoLMetrics {
    double algBwGBs;               // Algorithm bandwidth (bytes / time)
    double busBwGBs;               // Bus bandwidth (for 2 GPUs, same as algbw)
    double unidirectionalPeakGBs;  // Unidirectional NVLink peak (nvbandwidth reference)
    double bidirectionalPeakGBs;   // Bidirectional NVLink peak (theoretical max)
    double theoreticalMinUs;       // Minimum latency at unidirectional line rate
    double solPercent;             // SoL% relative to unidirectional (can exceed 100%)
    int nvlinkCount;               // Number of NVLinks detected
};

inline SoLMetrics CalculateSoL(size_t bytes, double avgSec, int numGpus, const NVLinkInfo& nvlink) {
    SoLMetrics sol = {};
    sol.nvlinkCount = nvlink.linkCount;

    // Algorithm bandwidth = data_size / time (NCCL convention)
    sol.algBwGBs = (static_cast<double>(bytes) / avgSec) / 1e9;

    // Bus bandwidth for ring AllReduce = algbw * (n-1)/n
    // For 2 GPUs: busbw = algbw (each GPU sends all data once)
    sol.busBwGBs = sol.algBwGBs;

    // Store both unidirectional and bidirectional peaks
    sol.unidirectionalPeakGBs = nvlink.unidirectionalGBs;
    sol.bidirectionalPeakGBs = nvlink.bidirectionalGBs;

    // Theoretical minimum latency at unidirectional line rate
    sol.theoreticalMinUs = (static_cast<double>(bytes) / (nvlink.unidirectionalGBs * 1e9)) * 1e6;

    // SoL% relative to unidirectional NVLink bandwidth
    // Values >100% indicate bidirectional utilization (expected for Yali)
    if (sol.unidirectionalPeakGBs > 0) {
        sol.solPercent = (sol.algBwGBs / sol.unidirectionalPeakGBs) * 100.0;
    } else {
        sol.solPercent = 0.0;
    }

    return sol;
}

}  // namespace yali
