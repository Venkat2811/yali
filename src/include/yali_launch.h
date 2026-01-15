/*************************************************************************
 * Copyright (c) 2025
 * All rights reserved.
 *
 * Yali kernel launch interface shared between host and device code.
 ************************************************************************/

#ifndef YALI_LAUNCH_H_
#define YALI_LAUNCH_H_

#include <stddef.h>
#include <stdint.h>

#include "collectives.h"

struct YaliLaunchArgs {
    // Outbound ring configuration.
    uint64_t* sendSequence;
    uint64_t* sendGating;
    char* sendData;
    int32_t sendCapacity;
    int32_t sendSlotBytes;
    int32_t sendSlotStride;

    // Inbound ring configuration.
    uint64_t* recvSequence;
    uint64_t* recvGating;
    char* recvData;
    int32_t recvCapacity;
    int32_t recvSlotBytes;
    int32_t recvSlotStride;

    // Local buffers.
    void* localInput;
    void* localOutput;
    void* peerInput;
    size_t elementCount;
    int elementSize;
    uint64_t sendOffset;
    uint64_t recvOffset;
    uint64_t initialSequence;

    ncclDataType_t datatype;
    ncclRedOp_t redOp;

    // Debug/bring-up options
    int debugEarlyExit;  // if non-zero, exit kernel after seed write
    int rank;            // caller rank for optional role-branching
    int worldSize;       // total number of ranks in the communicator

    // Additional metadata for multi-lane and low-latency variants.
    int laneIndex;    // 0-based lane identifier within the plan.
    int laneCount;    // total number of lanes participating.
    int ctasPerLane;  // grid decomposition used for the lane.
    int flash;        // non-zero when the low-latency kernel is active.
    int reserved0;
    int reserved1;
};

#endif  // YALI_LAUNCH_H_
