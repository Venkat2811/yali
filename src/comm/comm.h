/*************************************************************************
 * Yali Multiprocess Communicator
 *
 * Provides MPI-based bootstrap and synchronization for multi-process
 * AllReduce operations. Each process manages one GPU.
 *
 * Usage:
 *   YaliMPComm* comm = yaliMPCommCreate(&argc, &argv);
 *   cudaSetDevice(comm->localRank);
 *   // ... allocate buffers, exchange IPC handles ...
 *   yaliMPCommBarrier(comm);
 *   // ... launch kernels ...
 *   yaliMPCommDestroy(comm);
 ************************************************************************/

#ifndef YALI_COMM_H_
#define YALI_COMM_H_

#include <cuda_runtime.h>

#include <stdint.h>

#ifdef YALI_MPI_SUPPORT
    #include <mpi.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Multiprocess communicator state.
 *
 * In MPI mode: one process per GPU, rank == localRank for single-node.
 * In non-MPI mode: falls back to single-process behavior (worldSize=1).
 */
typedef struct YaliMPComm {
    int rank;       // Global rank (0 to worldSize-1)
    int worldSize;  // Total number of ranks
    int localRank;  // Local rank on this node (GPU index)
    int localSize;  // Number of ranks on this node
    int useMpi;     // Non-zero if MPI mode is active

    // IPC state (populated by yaliIpcExchange)
    cudaIpcMemHandle_t* peerHandles;  // [worldSize] IPC handles from peers
    void** peerPtrs;                  // [worldSize] Opened IPC pointers

    // CUDA resources
    cudaStream_t stream;

#ifdef YALI_MPI_SUPPORT
    MPI_Comm mpiComm;  // MPI communicator (typically MPI_COMM_WORLD)
#else
    void* mpiComm;  // Placeholder for ABI compatibility
#endif

    // Unique ID for IPC socket naming (if needed)
    uint64_t uniqueId;
} YaliMPComm;

/**
 * Create and initialize a multiprocess communicator.
 *
 * In MPI mode:
 *   - Calls MPI_Init if not already initialized
 *   - Discovers rank, worldSize, localRank via MPI
 *   - Sets up CUDA device based on localRank
 *
 * In non-MPI mode:
 *   - Returns a comm with rank=0, worldSize=1
 *
 * @param argc Pointer to argc from main()
 * @param argv Pointer to argv from main()
 * @return Allocated communicator (caller must call yaliMPCommDestroy)
 */
YaliMPComm* yaliMPCommCreate(int* argc, char*** argv);

/**
 * Destroy communicator and free resources.
 *
 * In MPI mode: Calls MPI_Finalize if this was the initializing call.
 *
 * @param comm Communicator to destroy (may be NULL)
 */
void yaliMPCommDestroy(YaliMPComm* comm);

/**
 * Barrier synchronization across all ranks.
 *
 * Synchronizes CUDA stream, then performs MPI_Barrier.
 *
 * @param comm Communicator
 */
void yaliMPCommBarrier(YaliMPComm* comm);

/**
 * Broadcast a value from rank 0 to all ranks.
 *
 * @param comm Communicator
 * @param data Pointer to data (input on rank 0, output on others)
 * @param bytes Size of data in bytes
 */
void yaliMPCommBroadcast(YaliMPComm* comm, void* data, size_t bytes);

/**
 * Get local rank from environment variables.
 *
 * Checks OMPI_COMM_WORLD_LOCAL_RANK (OpenMPI) and
 * MPI_LOCALRANKID (MPICH) environment variables.
 *
 * @return Local rank, or 0 if not found
 */
int yaliGetLocalRankFromEnv(void);

#ifdef __cplusplus
}
#endif

#endif  // YALI_COMM_H_
