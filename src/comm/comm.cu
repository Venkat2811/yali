/*************************************************************************
 * Yali Multiprocess Communicator - Implementation
 ************************************************************************/

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#include "comm.h"

// Helper macro to check MPI return codes (no return value, just log)
#define MPI_CHECK_LOG(call)                                                                                            \
    do {                                                                                                               \
        int mpi_ret = (call);                                                                                          \
        if (mpi_ret != MPI_SUCCESS) {                                                                                  \
            char mpi_err_str[MPI_MAX_ERROR_STRING];                                                                    \
            int mpi_err_len;                                                                                           \
            MPI_Error_string(mpi_ret, mpi_err_str, &mpi_err_len);                                                      \
            fprintf(stderr, "MPI error at %s:%d: %s\n", __FILE__, __LINE__, mpi_err_str);                              \
        }                                                                                                              \
    } while (0)

// Track whether we initialized MPI (so we know to finalize)
static int s_mpiInitializedByUs = 0;

int yaliGetLocalRankFromEnv(void) {
    // OpenMPI
    const char* ompi = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (ompi)
        return atoi(ompi);

    // MPICH / Intel MPI
    const char* mpich = getenv("MPI_LOCALRANKID");
    if (mpich)
        return atoi(mpich);

    // Slurm
    const char* slurm = getenv("SLURM_LOCALID");
    if (slurm)
        return atoi(slurm);

    // CUDA_VISIBLE_DEVICES single GPU case
    const char* cvd = getenv("CUDA_VISIBLE_DEVICES");
    if (cvd && strlen(cvd) == 1 && cvd[0] >= '0' && cvd[0] <= '9') {
        return 0;  // Single visible GPU, use device 0
    }

    return 0;
}

static int getLocalSizeFromEnv(void) {
    // OpenMPI
    const char* ompi = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
    if (ompi)
        return atoi(ompi);

    // Slurm
    const char* slurm = getenv("SLURM_NTASKS_PER_NODE");
    if (slurm)
        return atoi(slurm);

    return 1;
}

YaliMPComm* yaliMPCommCreate(int* argc, char*** argv) {
    YaliMPComm* comm = static_cast<YaliMPComm*>(calloc(1, sizeof(YaliMPComm)));
    if (!comm)
        return nullptr;

    // Initialize defaults
    comm->rank = 0;
    comm->worldSize = 1;
    comm->localRank = 0;
    comm->localSize = 1;
    comm->useMpi = 0;
    comm->peerHandles = nullptr;
    comm->peerPtrs = nullptr;
    comm->stream = nullptr;
    comm->mpiComm = nullptr;
    comm->uniqueId = static_cast<uint64_t>(getpid());

#ifdef YALI_MPI_SUPPORT
    // Check if MPI is already initialized
    int mpiInitialized = 0;
    MPI_Initialized(&mpiInitialized);

    if (!mpiInitialized) {
        int provided;
        MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
        s_mpiInitializedByUs = 1;

        if (provided < MPI_THREAD_MULTIPLE) {
            fprintf(stderr, "Warning: MPI_THREAD_MULTIPLE not supported (got %d)\n", provided);
        }
    }

    comm->mpiComm = MPI_COMM_WORLD;
    comm->useMpi = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &comm->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm->worldSize);

    // Get local rank via environment or MPI split
    comm->localRank = yaliGetLocalRankFromEnv();
    comm->localSize = getLocalSizeFromEnv();

    // If env vars not available, use MPI_Comm_split_type
    if (comm->localRank == 0 && comm->rank != 0) {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, comm->rank, MPI_INFO_NULL, &nodeComm);
        MPI_Comm_rank(nodeComm, &comm->localRank);
        MPI_Comm_size(nodeComm, &comm->localSize);
        MPI_Comm_free(&nodeComm);
    }

    // Generate unique ID for IPC naming (broadcast from rank 0)
    if (comm->rank == 0) {
        comm->uniqueId = static_cast<uint64_t>(getpid()) ^ (static_cast<uint64_t>(time(nullptr)) << 32);
    }
    MPI_CHECK_LOG(MPI_Bcast(&comm->uniqueId, sizeof(comm->uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));

#else
    // Non-MPI mode: single process
    (void)argc;
    (void)argv;
    comm->localRank = 0;
    comm->localSize = 1;
#endif

    // Set CUDA device based on local rank
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0 && comm->localRank < deviceCount) {
        cudaSetDevice(comm->localRank);
    }

    // Create CUDA stream
    cudaStreamCreate(&comm->stream);

    return comm;
}

void yaliMPCommDestroy(YaliMPComm* comm) {
    if (!comm)
        return;

    // Close IPC handles
    if (comm->peerPtrs) {
        for (int i = 0; i < comm->worldSize; i++) {
            if (i != comm->rank && comm->peerPtrs[i]) {
                cudaIpcCloseMemHandle(comm->peerPtrs[i]);
            }
        }
        free(comm->peerPtrs);
    }
    if (comm->peerHandles) {
        free(comm->peerHandles);
    }

    // Destroy CUDA stream
    if (comm->stream) {
        cudaStreamDestroy(comm->stream);
    }

#ifdef YALI_MPI_SUPPORT
    // Finalize MPI if we initialized it
    if (s_mpiInitializedByUs) {
        MPI_Finalize();
        s_mpiInitializedByUs = 0;
    }
#endif

    free(comm);
}

void yaliMPCommBarrier(YaliMPComm* comm) {
    if (!comm)
        return;

    // Sync CUDA stream first
    if (comm->stream) {
        cudaStreamSynchronize(comm->stream);
    }

#ifdef YALI_MPI_SUPPORT
    if (comm->useMpi) {
        MPI_CHECK_LOG(MPI_Barrier(comm->mpiComm));
    }
#endif
}

void yaliMPCommBroadcast(YaliMPComm* comm, void* data, size_t bytes) {
    if (!comm || !data || bytes == 0)
        return;

#ifdef YALI_MPI_SUPPORT
    if (comm->useMpi) {
        // Handle bytes > INT_MAX by chunking
        size_t offset = 0;
        while (offset < bytes) {
            size_t chunk = bytes - offset;
            if (chunk > static_cast<size_t>(INT_MAX)) {
                chunk = static_cast<size_t>(INT_MAX);
            }
            MPI_CHECK_LOG(
                MPI_Bcast(static_cast<char*>(data) + offset, static_cast<int>(chunk), MPI_BYTE, 0, comm->mpiComm));
            offset += chunk;
        }
    }
#else
    (void)data;
    (void)bytes;
#endif
}
