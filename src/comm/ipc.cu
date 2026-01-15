/*************************************************************************
 * Yali IPC Handle Management - Implementation
 ************************************************************************/

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#include "ipc.cuh"

// Helper macro to check MPI return codes
#define MPI_CHECK(call, cleanup)                                                                                       \
    do {                                                                                                               \
        int mpi_ret = (call);                                                                                          \
        if (mpi_ret != MPI_SUCCESS) {                                                                                  \
            char mpi_err_str[MPI_MAX_ERROR_STRING];                                                                    \
            int mpi_err_len;                                                                                           \
            MPI_Error_string(mpi_ret, mpi_err_str, &mpi_err_len);                                                      \
            fprintf(stderr, "MPI error at %s:%d: %s\n", __FILE__, __LINE__, mpi_err_str);                              \
            cleanup;                                                                                                   \
            return -1;                                                                                                 \
        }                                                                                                              \
    } while (0)

cudaError_t yaliIpcExport(void* devPtr, size_t size, YaliIpcHandle* handle) {
    if (!devPtr || !handle) {
        return cudaErrorInvalidValue;
    }

    memset(handle, 0, sizeof(YaliIpcHandle));
    handle->size = size;
    handle->srcRank = -1;  // Caller should set this

    return cudaIpcGetMemHandle(&handle->handle, devPtr);
}

cudaError_t yaliIpcImport(const YaliIpcHandle* handle, void** devPtr) {
    if (!handle || !devPtr) {
        return cudaErrorInvalidValue;
    }

    return cudaIpcOpenMemHandle(devPtr, handle->handle, cudaIpcMemLazyEnablePeerAccess);
}

cudaError_t yaliIpcClose(void* importedPtr) {
    if (!importedPtr) {
        return cudaSuccess;  // Closing NULL is a no-op
    }
    return cudaIpcCloseMemHandle(importedPtr);
}

int yaliIpcExchangeBuffers(YaliMPComm* comm, void* localBuf, size_t size) {
    if (!comm || !localBuf || size == 0) {
        return -1;
    }

#ifdef YALI_MPI_SUPPORT
    if (!comm->useMpi) {
        // Non-MPI mode: just store local buffer
        comm->peerPtrs = static_cast<void**>(calloc(1, sizeof(void*)));
        if (!comm->peerPtrs)
            return -1;
        comm->peerPtrs[0] = localBuf;
        return 0;
    }

    // Ensure all GPU work is complete before IPC handle export
    // IPC handles must be valid when exchanged - this prevents races
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        fprintf(stderr, "Rank %d: cudaDeviceSynchronize failed before IPC export: %s\n", comm->rank,
                cudaGetErrorString(syncErr));
        return -1;
    }

    // Export local buffer
    YaliIpcHandle myHandle;
    cudaError_t err = yaliIpcExport(localBuf, size, &myHandle);
    if (err != cudaSuccess) {
        fprintf(stderr, "Rank %d: Failed to export IPC handle: %s\n", comm->rank, cudaGetErrorString(err));
        return -1;
    }
    myHandle.srcRank = comm->rank;

    // Allocate arrays for all handles
    comm->peerHandles = static_cast<cudaIpcMemHandle_t*>(calloc(comm->worldSize, sizeof(cudaIpcMemHandle_t)));
    if (!comm->peerHandles) {
        fprintf(stderr, "Rank %d: Failed to allocate peerHandles array\n", comm->rank);
        return -1;
    }

    comm->peerPtrs = static_cast<void**>(calloc(comm->worldSize, sizeof(void*)));
    if (!comm->peerPtrs) {
        fprintf(stderr, "Rank %d: Failed to allocate peerPtrs array\n", comm->rank);
        free(comm->peerHandles);
        comm->peerHandles = nullptr;
        return -1;
    }

    // Allgather handles from all ranks - with error checking
    MPI_CHECK(MPI_Allgather(&myHandle.handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, comm->peerHandles,
                            sizeof(cudaIpcMemHandle_t), MPI_BYTE, comm->mpiComm),
              {
                  free(comm->peerPtrs);
                  free(comm->peerHandles);
                  comm->peerPtrs = nullptr;
                  comm->peerHandles = nullptr;
              });

    // Open handles from all peers
    for (int i = 0; i < comm->worldSize; i++) {
        if (i == comm->rank) {
            // Self: use local pointer directly (no IPC needed)
            comm->peerPtrs[i] = localBuf;
        } else {
            // Peer: open IPC handle
            err = cudaIpcOpenMemHandle(&comm->peerPtrs[i], comm->peerHandles[i], cudaIpcMemLazyEnablePeerAccess);
            if (err != cudaSuccess) {
                fprintf(stderr, "Rank %d: Failed to open IPC handle from rank %d: %s\n", comm->rank, i,
                        cudaGetErrorString(err));
                // Clean up already-opened handles
                for (int j = 0; j < i; j++) {
                    if (j != comm->rank && comm->peerPtrs[j]) {
                        cudaIpcCloseMemHandle(comm->peerPtrs[j]);
                    }
                }
                free(comm->peerPtrs);
                free(comm->peerHandles);
                comm->peerPtrs = nullptr;
                comm->peerHandles = nullptr;
                return -1;
            }
        }
    }

    return 0;

#else
    // Non-MPI mode: just store local buffer
    comm->peerPtrs = static_cast<void**>(calloc(1, sizeof(void*)));
    if (!comm->peerPtrs)
        return -1;
    comm->peerPtrs[0] = localBuf;
    return 0;
#endif
}

void yaliIpcCloseAll(YaliMPComm* comm) {
    if (!comm || !comm->peerPtrs)
        return;

    for (int i = 0; i < comm->worldSize; i++) {
        if (i != comm->rank && comm->peerPtrs[i]) {
            cudaIpcCloseMemHandle(comm->peerPtrs[i]);
            comm->peerPtrs[i] = nullptr;
        }
    }
}

// -----------------------------------------------------------------------------
// Ring Buffer IPC for Bandwidth Kernel
// -----------------------------------------------------------------------------

int yaliIpcExchangeRingBuffers(YaliMPComm* comm, const void* localRingsPtr, int numLanes, void*** peerSequence,
                               void*** peerGating, void*** peerData) {
    if (!comm || !localRingsPtr || numLanes <= 0) {
        return -1;
    }

    // Validate numLanes to prevent integer overflow in array indexing
    // worldSize * numLanes must fit in int32_t
    if (numLanes > (INT32_MAX / comm->worldSize)) {
        fprintf(stderr, "Rank %d: numLanes=%d would cause integer overflow with worldSize=%d\n", comm->rank, numLanes,
                comm->worldSize);
        return -1;
    }

#ifdef YALI_MPI_SUPPORT
    if (!comm->useMpi) {
        fprintf(stderr, "yaliIpcExchangeRingBuffers requires MPI mode\n");
        return -1;
    }

    // Ensure all GPU work is complete before IPC handle export
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        fprintf(stderr, "Rank %d: cudaDeviceSynchronize failed before ring IPC export: %s\n", comm->rank,
                cudaGetErrorString(syncErr));
        return -1;
    }

    // Cast to expected ring structure (matches ManagedRing in main_mpi.cu)
    struct RingInfo {
        uint64_t* sequence;
        uint64_t* gating;
        char* data;
        int capacity;
        size_t sequenceBytes;
        size_t dataBytes;
    };
    const RingInfo* localRings = static_cast<const RingInfo*>(localRingsPtr);

    // Allocate handle arrays
    YaliRingIpcHandles* myHandles = new (std::nothrow) YaliRingIpcHandles[numLanes];
    if (!myHandles) {
        fprintf(stderr, "Rank %d: Failed to allocate myHandles array\n", comm->rank);
        return -1;
    }

    const size_t allHandlesCount = static_cast<size_t>(comm->worldSize) * static_cast<size_t>(numLanes);
    YaliRingIpcHandles* allHandles = new (std::nothrow) YaliRingIpcHandles[allHandlesCount];
    if (!allHandles) {
        fprintf(stderr, "Rank %d: Failed to allocate allHandles array\n", comm->rank);
        delete[] myHandles;
        return -1;
    }

    // Export local handles for each lane
    for (int lane = 0; lane < numLanes; ++lane) {
        memset(&myHandles[lane], 0, sizeof(YaliRingIpcHandles));
        myHandles[lane].sequenceBytes = localRings[lane].sequenceBytes;
        myHandles[lane].gatingBytes = sizeof(uint64_t);
        myHandles[lane].dataBytes = localRings[lane].dataBytes;
        myHandles[lane].capacity = localRings[lane].capacity;

        if (localRings[lane].sequence && localRings[lane].gating && localRings[lane].data) {
            cudaError_t err;
            err = cudaIpcGetMemHandle(&myHandles[lane].sequenceHandle, localRings[lane].sequence);
            if (err != cudaSuccess) {
                fprintf(stderr, "Rank %d lane %d: Failed to export sequence handle: %s\n", comm->rank, lane,
                        cudaGetErrorString(err));
                delete[] myHandles;
                delete[] allHandles;
                return -1;
            }
            err = cudaIpcGetMemHandle(&myHandles[lane].gatingHandle, localRings[lane].gating);
            if (err != cudaSuccess) {
                fprintf(stderr, "Rank %d lane %d: Failed to export gating handle: %s\n", comm->rank, lane,
                        cudaGetErrorString(err));
                delete[] myHandles;
                delete[] allHandles;
                return -1;
            }
            err = cudaIpcGetMemHandle(&myHandles[lane].dataHandle, localRings[lane].data);
            if (err != cudaSuccess) {
                fprintf(stderr, "Rank %d lane %d: Failed to export data handle: %s\n", comm->rank, lane,
                        cudaGetErrorString(err));
                delete[] myHandles;
                delete[] allHandles;
                return -1;
            }
            myHandles[lane].valid = 1;
        } else {
            myHandles[lane].valid = 0;
        }
    }

    // Exchange all handles via MPI_Allgather - with error checking
    MPI_CHECK(MPI_Allgather(myHandles, numLanes * sizeof(YaliRingIpcHandles), MPI_BYTE, allHandles,
                            numLanes * sizeof(YaliRingIpcHandles), MPI_BYTE, comm->mpiComm),
              {
                  delete[] myHandles;
                  delete[] allHandles;
              });

    // Allocate output arrays
    *peerSequence = new (std::nothrow) void*[allHandlesCount];
    *peerGating = new (std::nothrow) void*[allHandlesCount];
    *peerData = new (std::nothrow) void*[allHandlesCount];

    if (!*peerSequence || !*peerGating || !*peerData) {
        fprintf(stderr, "Rank %d: Failed to allocate peer pointer arrays\n", comm->rank);
        delete[] *peerSequence;
        *peerSequence = nullptr;
        delete[] *peerGating;
        *peerGating = nullptr;
        delete[] *peerData;
        *peerData = nullptr;
        delete[] myHandles;
        delete[] allHandles;
        return -1;
    }

    // Initialize to null
    for (size_t i = 0; i < allHandlesCount; ++i) {
        (*peerSequence)[i] = nullptr;
        (*peerGating)[i] = nullptr;
        (*peerData)[i] = nullptr;
    }

    // Open peer handles - track opened handles for cleanup on failure
    bool openFailed = false;
    for (int r = 0; r < comm->worldSize && !openFailed; ++r) {
        for (int lane = 0; lane < numLanes && !openFailed; ++lane) {
            const size_t idx = static_cast<size_t>(r) * static_cast<size_t>(numLanes) + static_cast<size_t>(lane);

            if (r == comm->rank) {
                // Self: use local pointers directly (no IPC needed)
                (*peerSequence)[idx] = localRings[lane].sequence;
                (*peerGating)[idx] = localRings[lane].gating;
                (*peerData)[idx] = localRings[lane].data;
            } else {
                // Peer: import via IPC
                if (allHandles[idx].valid) {
                    cudaError_t err;
                    err = cudaIpcOpenMemHandle(&(*peerSequence)[idx], allHandles[idx].sequenceHandle,
                                               cudaIpcMemLazyEnablePeerAccess);
                    if (err != cudaSuccess) {
                        fprintf(stderr, "Rank %d: Failed to open sequence from rank %d lane %d: %s\n", comm->rank, r,
                                lane, cudaGetErrorString(err));
                        openFailed = true;
                        break;
                    }
                    err = cudaIpcOpenMemHandle(&(*peerGating)[idx], allHandles[idx].gatingHandle,
                                               cudaIpcMemLazyEnablePeerAccess);
                    if (err != cudaSuccess) {
                        fprintf(stderr, "Rank %d: Failed to open gating from rank %d lane %d: %s\n", comm->rank, r,
                                lane, cudaGetErrorString(err));
                        // Close sequence we just opened
                        cudaIpcCloseMemHandle((*peerSequence)[idx]);
                        (*peerSequence)[idx] = nullptr;
                        openFailed = true;
                        break;
                    }
                    err = cudaIpcOpenMemHandle(&(*peerData)[idx], allHandles[idx].dataHandle,
                                               cudaIpcMemLazyEnablePeerAccess);
                    if (err != cudaSuccess) {
                        fprintf(stderr, "Rank %d: Failed to open data from rank %d lane %d: %s\n", comm->rank, r, lane,
                                cudaGetErrorString(err));
                        // Close sequence and gating we just opened
                        cudaIpcCloseMemHandle((*peerSequence)[idx]);
                        cudaIpcCloseMemHandle((*peerGating)[idx]);
                        (*peerSequence)[idx] = nullptr;
                        (*peerGating)[idx] = nullptr;
                        openFailed = true;
                        break;
                    }
                }
            }
        }
    }

    if (openFailed) {
        // Clean up any handles we successfully opened before failure
        for (int r = 0; r < comm->worldSize; ++r) {
            if (r == comm->rank)
                continue;
            for (int lane = 0; lane < numLanes; ++lane) {
                const size_t idx = static_cast<size_t>(r) * static_cast<size_t>(numLanes) + static_cast<size_t>(lane);
                if ((*peerSequence)[idx])
                    cudaIpcCloseMemHandle((*peerSequence)[idx]);
                if ((*peerGating)[idx])
                    cudaIpcCloseMemHandle((*peerGating)[idx]);
                if ((*peerData)[idx])
                    cudaIpcCloseMemHandle((*peerData)[idx]);
            }
        }
        delete[] *peerSequence;
        *peerSequence = nullptr;
        delete[] *peerGating;
        *peerGating = nullptr;
        delete[] *peerData;
        *peerData = nullptr;
        delete[] myHandles;
        delete[] allHandles;
        return -1;
    }

    delete[] myHandles;
    delete[] allHandles;
    return 0;

#else
    fprintf(stderr, "yaliIpcExchangeRingBuffers requires YALI_MPI_SUPPORT\n");
    return -1;
#endif
}

void yaliIpcCloseRingBuffers(YaliMPComm* comm, int numLanes, void** peerSequence, void** peerGating, void** peerData) {
    if (!comm)
        return;

    for (int r = 0; r < comm->worldSize; ++r) {
        if (r == comm->rank)
            continue;  // Don't close self pointers

        for (int lane = 0; lane < numLanes; ++lane) {
            int idx = r * numLanes + lane;

            if (peerSequence && peerSequence[idx]) {
                cudaIpcCloseMemHandle(peerSequence[idx]);
            }
            if (peerGating && peerGating[idx]) {
                cudaIpcCloseMemHandle(peerGating[idx]);
            }
            if (peerData && peerData[idx]) {
                cudaIpcCloseMemHandle(peerData[idx]);
            }
        }
    }

    delete[] peerSequence;
    delete[] peerGating;
    delete[] peerData;
}
