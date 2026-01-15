/*************************************************************************
 * Yali IPC Handle Management
 *
 * Provides CUDA IPC handle export/import for cross-process buffer sharing.
 * Used by MPI-mode harness to share GPU memory between processes.
 *
 * Usage:
 *   // On each rank: export local buffer
 *   YaliIpcHandle handle;
 *   yaliIpcExport(myBuffer, size, &handle);
 *
 *   // Exchange handles via MPI (see yaliIpcExchangeBuffers)
 *
 *   // Import peer's buffer
 *   void* peerPtr;
 *   yaliIpcImport(&peerHandle, &peerPtr);
 ************************************************************************/

#ifndef YALI_IPC_CUH_
#define YALI_IPC_CUH_

#include <cuda_runtime.h>

#include <stddef.h>

#include "comm.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * IPC handle with metadata.
 */
typedef struct YaliIpcHandle {
    cudaIpcMemHandle_t handle;  // CUDA IPC handle
    size_t size;                // Buffer size in bytes
    int srcRank;                // Source rank that exported this handle
} YaliIpcHandle;

/**
 * Export a device buffer for cross-process access.
 *
 * The buffer must be allocated with cudaMalloc (not cudaMallocManaged).
 *
 * @param devPtr Device pointer to export
 * @param size Size of the buffer in bytes
 * @param handle Output: IPC handle to share with peers
 * @return cudaSuccess on success
 */
cudaError_t yaliIpcExport(void* devPtr, size_t size, YaliIpcHandle* handle);

/**
 * Import a peer's buffer using their IPC handle.
 *
 * The returned pointer is valid for GPU access in this process.
 * Must be closed with yaliIpcClose when done.
 *
 * @param handle IPC handle from peer
 * @param devPtr Output: device pointer usable in this process
 * @return cudaSuccess on success
 */
cudaError_t yaliIpcImport(const YaliIpcHandle* handle, void** devPtr);

/**
 * Close an imported IPC handle.
 *
 * @param importedPtr Pointer returned by yaliIpcImport
 * @return cudaSuccess on success
 */
cudaError_t yaliIpcClose(void* importedPtr);

/**
 * Exchange buffer IPC handles between all ranks.
 *
 * After this call:
 *   - comm->peerHandles[i] contains the IPC handle from rank i
 *   - comm->peerPtrs[i] contains the opened pointer to rank i's buffer
 *   - comm->peerPtrs[comm->rank] == localBuf (no IPC needed for self)
 *
 * @param comm Communicator (must be MPI-enabled)
 * @param localBuf Local device buffer to share
 * @param size Size of the buffer in bytes
 * @return 0 on success, -1 on error
 */
int yaliIpcExchangeBuffers(YaliMPComm* comm, void* localBuf, size_t size);

/**
 * Close all imported IPC handles in the communicator.
 *
 * Called automatically by yaliMPCommDestroy.
 *
 * @param comm Communicator
 */
void yaliIpcCloseAll(YaliMPComm* comm);

// -----------------------------------------------------------------------------
// Ring Buffer IPC for Bandwidth Kernel
// -----------------------------------------------------------------------------

/**
 * Ring buffer IPC handles for stream kernel.
 * Each lane has its own ring buffer (sequence, gating, data).
 */
typedef struct YaliRingIpcHandles {
    cudaIpcMemHandle_t sequenceHandle;
    cudaIpcMemHandle_t gatingHandle;
    cudaIpcMemHandle_t dataHandle;
    size_t sequenceBytes;
    size_t gatingBytes;
    size_t dataBytes;
    int capacity;
    int valid;  // 1 if handles are valid (non-null buffers)
} YaliRingIpcHandles;

/**
 * Exchange ring buffer IPC handles for all lanes.
 *
 * For stream kernel: each rank exports its ring buffers (sequence, gating, data)
 * and imports peer's ring buffers for cross-process access.
 *
 * @param comm MPI communicator
 * @param localRings Local ring buffers (array of ManagedRing-like structs)
 * @param numLanes Number of lanes
 * @param peerSequence Output: [worldSize * numLanes] sequence pointers
 * @param peerGating Output: [worldSize * numLanes] gating pointers
 * @param peerData Output: [worldSize * numLanes] data pointers
 * @return 0 on success, -1 on error
 */
int yaliIpcExchangeRingBuffers(YaliMPComm* comm, const void* localRings, int numLanes, void*** peerSequence,
                               void*** peerGating, void*** peerData);

/**
 * Close ring buffer IPC handles opened by yaliIpcExchangeRingBuffers.
 *
 * @param comm Communicator
 * @param numLanes Number of lanes
 * @param peerSequence Sequence pointers to close
 * @param peerGating Gating pointers to close
 * @param peerData Data pointers to close
 */
void yaliIpcCloseRingBuffers(YaliMPComm* comm, int numLanes, void** peerSequence, void** peerGating, void** peerData);

#ifdef __cplusplus
}
#endif

#endif  // YALI_IPC_CUH_
