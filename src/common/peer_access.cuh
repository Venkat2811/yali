// Yali peer access utilities
// Provides P2P (peer-to-peer) memory access setup between GPUs
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace yali {

// Result of P2P check
struct P2PCapability {
    int device0;
    int device1;
    bool can_access_0_to_1;
    bool can_access_1_to_0;
    bool bidirectional;
};

// Check if P2P access is possible between two devices
inline P2PCapability CheckP2PAccess(int dev0, int dev1) {
    P2PCapability cap = {};
    cap.device0 = dev0;
    cap.device1 = dev1;

    int canAccess01 = 0, canAccess10 = 0;
    cudaError_t err0 = cudaDeviceCanAccessPeer(&canAccess01, dev0, dev1);
    cudaError_t err1 = cudaDeviceCanAccessPeer(&canAccess10, dev1, dev0);

    cap.can_access_0_to_1 = (err0 == cudaSuccess && canAccess01 != 0);
    cap.can_access_1_to_0 = (err1 == cudaSuccess && canAccess10 != 0);
    cap.bidirectional = cap.can_access_0_to_1 && cap.can_access_1_to_0;

    return cap;
}

// Enable P2P access from dev to peer
// Returns cudaSuccess on success, or error code on failure
inline cudaError_t EnableP2PAccess(int dev, int peer) {
    int canAccess = 0;
    cudaError_t err = cudaDeviceCanAccessPeer(&canAccess, dev, peer);
    if (err != cudaSuccess) {
        return err;
    }
    if (!canAccess) {
        return cudaErrorPeerAccessUnsupported;
    }

    err = cudaSetDevice(dev);
    if (err != cudaSuccess) {
        return err;
    }

    err = cudaDeviceEnablePeerAccess(peer, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        // Clear the error and treat as success
        cudaGetLastError();
        return cudaSuccess;
    }
    return err;
}

// Enable bidirectional P2P access between two devices
// Returns true if both directions enabled successfully
inline bool EnableBidirectionalP2P(int dev0, int dev1) {
    cudaError_t err0 = EnableP2PAccess(dev0, dev1);
    cudaError_t err1 = EnableP2PAccess(dev1, dev0);
    return (err0 == cudaSuccess && err1 == cudaSuccess);
}

// Enable P2P access and exit on failure (harness-style)
inline void EnablePeerAccessOrDie(int dev, int peer) {
    int canAccess = 0;
    cudaError_t err = cudaDeviceCanAccessPeer(&canAccess, dev, peer);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceCanAccessPeer failed: %s\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
    if (!canAccess) {
        fprintf(stderr, "Device %d cannot access peer %d; enable NVLink/P2P for Yali harness.\n", dev, peer);
        std::exit(EXIT_FAILURE);
    }
    err = cudaSetDevice(dev);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", dev, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
    err = cudaDeviceEnablePeerAccess(peer, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
    } else if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceEnablePeerAccess(%d, %d) failed: %s\n", dev, peer, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

// Query topology information
struct TopologyInfo {
    int device_count;
    std::vector<P2PCapability> p2p_capabilities;  // All pairs

    bool AllP2PEnabled() const {
        for (const auto& cap : p2p_capabilities) {
            if (!cap.bidirectional)
                return false;
        }
        return true;
    }
};

inline TopologyInfo QueryTopology() {
    TopologyInfo info = {};
    cudaGetDeviceCount(&info.device_count);

    for (int i = 0; i < info.device_count; ++i) {
        for (int j = i + 1; j < info.device_count; ++j) {
            info.p2p_capabilities.push_back(CheckP2PAccess(i, j));
        }
    }
    return info;
}

// Print topology summary
inline void PrintTopology(const TopologyInfo& info) {
    printf("Device count: %d\n", info.device_count);
    for (const auto& cap : info.p2p_capabilities) {
        printf("GPU%d <-> GPU%d: %s\n", cap.device0, cap.device1,
               cap.bidirectional                                  ? "P2P enabled"
               : (cap.can_access_0_to_1 || cap.can_access_1_to_0) ? "Partial P2P"
                                                                  : "No P2P");
    }
}

}  // namespace yali
