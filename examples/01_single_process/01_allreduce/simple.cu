/*************************************************************************
 * Simple AllReduce Example
 *
 * Minimal 2-GPU AllReduce using the high-level yali::allreduce API.
 * This is the recommended starting point for most users.
 *
 * Build:   bazel build //examples/01_single_process/01_allreduce:simple
 * Run:     CUDA_VISIBLE_DEVICES=0,1 bazel-bin/examples/01_single_process/01_allreduce/simple
 *
 * Features:
 *   - yali::Comm - Communicator with P2P setup
 *   - yali::allreduce() - Auto-tuned kernel selection
 *   - Separate send/recv buffers (NCCL-style API)
 ************************************************************************/

#include <cuda_runtime.h>

#include <cstdio>

#include "src/ops/allreduce.cuh"

int main() {
    // 1. Setup: create communicator for GPU 0 and 1
    yali::Comm comm(0, 1);
    if (!comm.ok()) {
        printf("P2P init failed\n");
        return 1;
    }

    // 2. Allocate send/recv buffers (1M floats per GPU)
    constexpr size_t N = 1024 * 1024;
    float *send0, *recv0, *send1, *recv1;

    cudaSetDevice(0);
    cudaMalloc(&send0, N * sizeof(float));
    cudaMalloc(&recv0, N * sizeof(float));

    cudaSetDevice(1);
    cudaMalloc(&send1, N * sizeof(float));
    cudaMalloc(&recv1, N * sizeof(float));

    // Initialize: GPU0 send = 1.0, GPU1 send = 2.0
    float one = 1.0f, two = 2.0f;
    cudaSetDevice(0);
    cudaMemset(send0, 0, N * sizeof(float));
    cudaMemcpy(send0, &one, sizeof(float), cudaMemcpyHostToDevice);

    cudaSetDevice(1);
    cudaMemset(send1, 0, N * sizeof(float));
    cudaMemcpy(send1, &two, sizeof(float), cudaMemcpyHostToDevice);

    // 3. AllReduce: recv = send0 + send1
    cudaError_t err = yali::allreduce(comm, send0, recv0, send1, recv1, N);
    if (err != cudaSuccess) {
        printf("AllReduce failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // 4. Verify: both recv buffers should have 3.0 at index 0
    float result0, result1;
    cudaSetDevice(0);
    cudaMemcpy(&result0, recv0, sizeof(float), cudaMemcpyDeviceToHost);
    cudaSetDevice(1);
    cudaMemcpy(&result1, recv1, sizeof(float), cudaMemcpyDeviceToHost);

    printf("GPU0[0]=%.1f, GPU1[0]=%.1f (expected: 3.0, 3.0)\n", result0, result1);

    cudaSetDevice(0);
    cudaFree(send0);
    cudaFree(recv0);
    cudaSetDevice(1);
    cudaFree(send1);
    cudaFree(recv1);
    return (result0 == 3.0f && result1 == 3.0f) ? 0 : 1;
}
