<p align="center">
  <img src="../assets/yali-icon.png" alt="YALI" width="200">
</p>

# YALI Architecture

This document explains how YALI achieves superior AllReduce performance through bidirectional NVLink utilization, async prefetch pipelines, and ring buffer flow control.

## Table of Contents

1. [Why YALI is Faster](#why-yali-is-faster)
2. [Two Kernel Modes](#two-kernel-modes)
3. [Flash Kernel Deep Dive](#flash-kernel-deep-dive)
4. [Stream Kernel Deep Dive](#stream-kernel-deep-dive)
5. [Memory Ordering](#memory-ordering)
6. [Code Map](#code-map)
7. [Examples](#examples)
8. [Benchmarks](#benchmarks)

---

## Why YALI is Faster

### NCCL Ring Algorithm (Sequential, Unidirectional)

NCCL's ring algorithm processes chunks sequentially, using one NVLink direction at a time:

```
NCCL Ring AllReduce (2 GPUs)
============================

Step 1: Scatter-Reduce (2 substeps)
-----------------------------------
  GPU0 has: [A0, A1]     GPU1 has: [B0, B1]

  Substep 1: GPU0 sends A1 ──────────────> GPU1 receives, computes B1 += A1
                           (0→1 direction)

  Substep 2: GPU1 sends B0 <────────────── GPU0 receives, computes A0 += B0
                           (1→0 direction)

Step 2: Allgather (2 substeps)
------------------------------
  Substep 1: GPU0 sends A0 ──────────────> GPU1 receives A0
                           (0→1 direction)

  Substep 2: GPU1 sends B1 <────────────── GPU0 receives B1
                           (1→0 direction)

Result: Both GPUs have [A0+B0, A1+B1]

NVLink Usage: ONE DIRECTION AT A TIME → max ~78% of unidirectional SoL
```

### YALI Algorithm (Simultaneous, Bidirectional)

YALI has both GPUs read from each other simultaneously, utilizing both NVLink directions:

```
YALI AllReduce (2 GPUs)
=======================

SIMULTANEOUSLY:
  GPU0 kernel: reads ALL of B from GPU1 ──┐
                                          ├──> BOTH NVLink directions active!
  GPU1 kernel: reads ALL of A from GPU0 ──┘

Each GPU:
  1. Reads peer data into shared memory (via NVLink)
  2. Loads local data into registers
  3. Reduces: result = local + peer
  4. Stores result to local buffer

Result: Both GPUs have A + B

NVLink Usage: BOTH DIRECTIONS SIMULTANEOUSLY → can achieve 87%+ SoL
```

### Performance Comparison

Benchmarked on 2x A100-SXM4-80GB (NVLink NV2, ~47 GB/s unidirectional):

```
+----------------+------------------+------------------+---------+
|                | NCCL Ring        | YALI             | Winner  |
+----------------+------------------+------------------+---------+
| Algorithm      | Scatter + Gather | Direct Read+Sum  | YALI    |
| NVLink Usage   | Unidirectional   | Bidirectional    | YALI    |
| Max SoL        | ~78%             | ~95%             | YALI    |
| Peak @ 2GB     | 37 GB/s          | 44 GB/s          | YALI    |
| Speedup @ 1MB  | -                | 1.85x            | YALI    |
+----------------+------------------+------------------+---------+
```

---

## Two Kernel Modes

YALI automatically selects the optimal kernel based on message size:

```
Message Size Selection
======================

  0          64 MB                                    ∞
  |-----------|-------------------------------------|
  |   FLASH   |              STREAM                 |
  |  kernel   |              kernel                 |
  |-----------|-------------------------------------|

  Flash: cp.async prefetch, multi-CTA, low latency
  Stream: Ring buffer, sequence flow control, high bandwidth
```

### Selection Heuristics

| Message Size | Kernel | Why |
|:-------------|:-------|:----|
| ≤ 64 MB | Flash | Lower launch overhead, prefetch hides latency |
| > 64 MB | Stream | Ring buffer prevents producer overrun |

---

## Flash Kernel Deep Dive

The Flash kernel uses CUDA's `cp.async` instruction for non-blocking memory transfers with a 3-stage prefetch pipeline:

```
Flash Kernel Pipeline (3 stages)
================================

Time →
Stage 0    Stage 1    Stage 2    Stage 0    Stage 1    ...
   |          |          |          |          |
   v          v          v          v          v
┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
│FETCH │  │FETCH │  │FETCH │  │FETCH │  │FETCH │
│peer  │  │peer  │  │peer  │  │peer  │  │peer  │
│chunk0│  │chunk1│  │chunk2│  │chunk3│  │chunk4│
└──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘
   │         │         │         │         │
   └────┐    └────┐    └────┐    └────┐    │
        │         │         │         │    │
        v         v         v         v    v
     ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
     │REDUCE│  │REDUCE│  │REDUCE│  │REDUCE│  ...
     │chunk0│  │chunk1│  │chunk2│  │chunk3│
     │+store│  │+store│  │+store│  │+store│
     └──────┘  └──────┘  └──────┘  └──────┘

Key: Fetch of chunk N+1 overlaps with reduce of chunk N
     → Memory latency is hidden
```

### cp.async Instruction

```cpp
// PTX: Non-blocking copy from global to shared memory
cp.async.cg.shared.global [smem_ptr], [gmem_ptr], 16;  // 16 bytes
cp.async.commit_group;   // Commit pending copies
cp.async.wait_group<1>;  // Wait for all but 1 group (pipelining)
```

### Flash Kernel Data Flow

```
GPU0                                    GPU1
====                                    ====

Local Buffer A                          Local Buffer B
┌─────────────┐                        ┌─────────────┐
│ A0 A1 A2 A3 │                        │ B0 B1 B2 B3 │
└─────────────┘                        └─────────────┘
       │                                      │
       │                                      │
       ▼                                      ▼
  ┌─────────┐      NVLink (read B)      ┌─────────┐
  │ Flash   │ ◄──────────────────────── │ Buffer  │
  │ Kernel  │                           │   B     │
  │         │ ──────────────────────► ◄─┤         │
  └────┬────┘      NVLink (read A)      └────┬────┘
       │                                      │
       │ A + B                                │ A + B
       ▼                                      ▼
Result Buffer                          Result Buffer
┌─────────────┐                        ┌─────────────┐
│ A+B  A+B... │                        │ A+B  A+B... │
└─────────────┘                        └─────────────┘
```

---

## Stream Kernel Deep Dive

For large messages (>64MB), the Stream kernel uses ring buffers with sequence-based flow control:

```
Stream Kernel Ring Buffer
=========================

GPU0 Ring Buffer (slots 0-7)           GPU1 Ring Buffer (slots 0-7)
┌───┬───┬───┬───┬───┬───┬───┬───┐     ┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │     │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │
└───┴───┴───┴───┴───┴───┴───┴───┘     └───┴───┴───┴───┴───┴───┴───┴───┘
  ▲                       ▲             ▲                       ▲
  │                       │             │                       │
  producer_seq=5          │             producer_seq=5          │
                    consumer_seq=2                        consumer_seq=2

Sequence Numbers:
- producer_seq: Next slot to be written
- consumer_seq: Next slot to be consumed (gating)
- Slot available when: slot_seq < producer_seq - NUM_SLOTS
```

### Ring Buffer Flow Control

```
Producer-Consumer Protocol
==========================

GPU0 (Producer)                        GPU1 (Consumer)
===============                        ===============

1. Check gating:                       1. Wait for data:
   wait until                             wait until
   consumer_seq > slot - NUM_SLOTS        producer_seq > slot

2. Write data to slot                  2. Read data from slot
                                          (via NVLink)

3. Publish:                            3. Reduce with local data
   __threadfence_system()
   producer_seq = slot + 1             4. Advance consumer:
                                          __threadfence_system()
                                          consumer_seq = slot + 1
```

### Multi-Lane Parallelism

For very large messages, the Stream kernel splits work across multiple independent lanes:

```
Multi-Lane Stream Kernel (4 lanes)
==================================

Total Data: 2 GB
            ┌─────────────────────────────────────────────────────┐
            │                    2 GB payload                     │
            └─────────────────────────────────────────────────────┘
                    │           │           │           │
                    ▼           ▼           ▼           ▼
            ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐
            │  Lane 0   │ │  Lane 1   │ │  Lane 2   │ │  Lane 3   │
            │  512 MB   │ │  512 MB   │ │  512 MB   │ │  512 MB   │
            │  Stream0  │ │  Stream1  │ │  Stream2  │ │  Stream3  │
            └───────────┘ └───────────┘ └───────────┘ └───────────┘
                    │           │           │           │
                    └───────────┴───────────┴───────────┘
                                    │
                                    ▼
                    All lanes execute in parallel on NVLink
```

---

## Memory Ordering

YALI uses acquire-release semantics for cross-GPU synchronization:

```
Acquire-Release Protocol
========================

Producer (GPU0)                        Consumer (GPU1)
===============                        ===============

// Write data to shared slot           // Wait for producer
slot_data[idx] = value;
...                                    while (producer_seq <= slot)
__threadfence_system();  // RELEASE      volatile_load(&producer_seq);
volatile_store(&producer_seq,          __threadfence_system();  // ACQUIRE
               slot + 1);
                                       // Now safe to read slot_data
                                       result = slot_data[idx];

Key: __threadfence_system() ensures writes are visible across GPUs
     Volatile loads/stores prevent compiler reordering
```

---

## Code Map

```
src/
├── include/                 # Public API
│   ├── yali.h              # Types: Dtype, Result
│   ├── yali_launch.h       # Launch args struct
│   └── yali_tuning.h       # Auto-tuning heuristics
│
├── kernels/                 # CUDA kernel implementations
│   ├── stream.cu           # Stream kernel (large messages)
│   ├── stream.cuh          # Stream kernel header
│   ├── ring.cuh            # Ring buffer primitives
│   └── type_ops.cuh        # FP32/FP16/BF16 operations
│
├── all_reduce/             # AllReduce interface
│   ├── all_reduce.h        # Internal API
│   ├── kernels.cuh         # Kernel dispatch
│   ├── stream.cuh          # Stream kernel wrapper
│   └── flash.cuh           # Flash kernel wrapper
│
├── ops/                    # User-facing API
│   ├── allreduce.cuh       # Single-process API
│   └── allreduce_mpi.cuh   # MPI multi-process API
│
└── comm/                   # Communication layer
    ├── comm.h              # MPI communicator
    └── ipc.cuh             # CUDA IPC handle exchange
```

---

## Examples

### Minimal Example (3 lines)

```cpp
#include "src/ops/allreduce.cuh"

// Setup (once)
yali::Comm comm(0, 1);  // GPU 0 and 1

// AllReduce
yali::allreduce(comm, send0, recv0, send1, recv1, count);
```

See: `examples/01_single_process/01_allreduce/simple.cu`

### MPI Example

```cpp
#include "src/ops/allreduce_mpi.cuh"

// MPI init creates comm automatically
yali::MPIComm comm;

// Each rank provides its local buffers
yali::allreduce_mpi(comm, my_send, my_recv, count);
```

See: `examples/02_multi_process/01_allreduce/simple_mpi.cu`

---

## Benchmarks

### Running Benchmarks

```bash
# Quick comparison
make bench                  # Single-process
make bench-mpi              # MPI mode

# Full sweep with statistics
make sweep                  # All dtypes (FP32, FP16, BF16)
make sweep-quick            # FP32 only, both modes
```

### Understanding Results

```
Speed of Light (SoL) Calculation
================================

SoL % = measured_bandwidth / nvbandwidth_D2D_unidir × 100

Example (A100 NV2):
  nvbandwidth D2D unidirectional: 47 GB/s
  YALI measured: 44 GB/s
  SoL = 44 / 47 × 100 = 94%

Why not 100% SoL:
  - Reduction ALU operations add overhead
  - Memory access patterns not perfectly coalesced
  - Synchronization barriers between stages
```

### Timing Modes

```
+-------------+------------------------------+------------------+
| Mode        | Description                  | Use Case         |
+-------------+------------------------------+------------------+
| cuda-events | GPU-only timing via events   | Fair comparison  |
| throughput  | Wall-clock, fire-and-forget  | Production-like  |
| latency     | Wall-clock, sync each call   | Interactive/BS=1 |
+-------------+------------------------------+------------------+
```

### Benchmark Reports

See `docs/benchmark/artifacts/` for full benchmark reports with graphs:

- [Standard Sweep (FP32/FP16/BF16)](benchmark/artifacts/2026-01-15/151357-standard/summary.md)
- [Extensive Sweep (all dtypes, more sizes)](benchmark/artifacts/2026-01-15/152933-extensive/summary.md)
- [Quick Sweep with Profiler (nsys)](benchmark/artifacts/2026-01-15/164632-quick-profiler/summary.md)

---

## Summary

YALI achieves superior performance through:

1. **Bidirectional NVLink** - Both GPUs read simultaneously
2. **cp.async prefetch** - Non-blocking memory transfers hide latency
3. **Ring buffer flow control** - Sequence-based synchronization without locks
4. **Multi-lane parallelism** - Large messages split across independent lanes
5. **Acquire-release semantics** - Correct ordering across GPUs with minimal overhead
