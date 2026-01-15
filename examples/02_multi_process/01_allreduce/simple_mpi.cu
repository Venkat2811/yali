/*************************************************************************
 * Simple MPI AllReduce Example
 *
 * Minimal 2-process AllReduce using the high-level yali::allreduce API.
 * This is the recommended starting point for MPI users.
 *
 * Build:   bazel build //:example_simple_mpi
 * Run:     CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 --allow-run-as-root bazel-bin/example_simple_mpi
 *
 * Features:
 *   - yali::MPIComm - MPI communicator with IPC setup
 *   - yali::allreduce() - Auto-tuned kernel selection
 *   - Single-rank buffer management (each rank manages its own buffers)
 ************************************************************************/

#include <cuda_runtime.h>

#include <cstdio>

#include "src/ops/allreduce_mpi.cuh"

int main(int argc, char** argv) {
    // 1. Setup: create MPI communicator (handles MPI_Init internally)
    yali::MPIComm comm(&argc, &argv);
    if (!comm.ok()) {
        printf("MPI init failed\n");
        return 1;
    }

    const int rank = comm.rank();

    // 2. Allocate send/recv buffers (1M floats on local GPU)
    constexpr size_t N = 1024 * 1024;
    float *send, *recv;

    cudaMalloc(&send, N * sizeof(float));
    cudaMalloc(&recv, N * sizeof(float));

    // Initialize: rank 0 send = 1.0, rank 1 send = 2.0
    float seedValue = static_cast<float>(rank + 1);
    cudaMemset(send, 0, N * sizeof(float));
    cudaMemcpy(send, &seedValue, sizeof(float), cudaMemcpyHostToDevice);

    if (rank == 0) {
        printf("=== Yali MPI AllReduce Example (ops API) ===\n");
        printf("World size: %d\n", comm.world_size());
        printf("Elements: %zu (%.2f MB)\n", N, N * sizeof(float) / 1e6);
    }

    // 3. AllReduce: recv = send_rank0 + send_rank1
    cudaError_t err = yali::allreduce(comm, send, recv, N);
    if (err != cudaSuccess) {
        printf("Rank %d: AllReduce failed: %s\n", rank, cudaGetErrorString(err));
        return 1;
    }

    // 4. Verify: both ranks should have 3.0 at index 0
    float result;
    cudaMemcpy(&result, recv, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Rank %d: recv[0]=%.1f (expected: 3.0)\n", rank, result);

    cudaFree(send);
    cudaFree(recv);

    bool passed = (result == 3.0f);
    if (rank == 0) {
        printf("=== Example %s ===\n", passed ? "PASSED" : "FAILED");
    }

    return passed ? 0 : 1;
}
