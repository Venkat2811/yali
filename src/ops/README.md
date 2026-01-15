# Yali Ops

High-level operations API for Yali. Simple, user-friendly wrappers around the kernel primitives.

## Available Operations

### `allreduce.cuh` - 2-GPU AllReduce (Single Process)

```cpp
#include "src/ops/allreduce.cuh"

// Initialize communicator (enables P2P)
yali::Comm comm(0, 1);
if (!comm.ok()) { /* handle error */ }

// AllReduce: reads from send buffers, writes sum to recv buffers
yali::allreduce(comm, send0, recv0, send1, recv1, count);
```

**Supported types:** `float`, `__half`, `__nv_bfloat16`

**API:** Uses separate send/recv buffers (like NCCL). The kernel reads from send buffers and writes the reduced result to recv buffers.

**Automatic kernel selection:** Uses low-latency kernel for small messages (<64MB).

### `allreduce_mpi.cuh` - 2-Process AllReduce (MPI)

```cpp
#include "src/ops/allreduce_mpi.cuh"

// Initialize MPI communicator (handles MPI_Init, IPC setup)
yali::MPIComm comm(&argc, &argv);
if (!comm.ok()) { /* handle error */ }

// Each rank manages its own buffers
float *send, *recv;
cudaMalloc(&send, count * sizeof(float));
cudaMalloc(&recv, count * sizeof(float));

// AllReduce: each rank contributes, all receive sum
yali::allreduce(comm, send, recv, count);
```

**Supported types:** `float`, `__half`, `__nv_bfloat16`

**API:** Each MPI rank manages its own local buffers. IPC handles are exchanged automatically on each call for correctness.

**Note:** For peak performance with stable buffers (same buffer used repeatedly), use the raw harness API which can cache IPC handles.

## Design Philosophy

1. **Minimal boilerplate** - 3 lines to AllReduce
2. **Sensible defaults** - Auto-tuned lane count, kernel mode
3. **Zero hidden state** - Comm holds only GPU indices (single-process) or MPI context (MPI)
4. **Header-only** - No separate compilation needed
5. **Correctness first** - Safe defaults over micro-optimizations

## Performance

Benchmarked on 2x A100-SXM4-80GB (NVLink NV2, ~47 GB/s unidirectional).

### Single-Process (allreduce.cuh)

The ops API achieves identical performance to the raw kernel harness:

| Size | YALI | NCCL | Speedup |
|:-----|:----:|:----:|:-------:|
| 1MB  | 27 GB/s | 14 GB/s | **1.85x** |
| 64MB | 39 GB/s | 34 GB/s | **1.15x** |
| 2GB  | 44 GB/s | 37 GB/s | **1.22x** |

### Multi-Process MPI (allreduce_mpi.cuh)

The MPI ops API achieves similar performance when using cached IPC handles:

| Size | YALI | NCCL | Speedup |
|:-----|:----:|:----:|:-------:|
| 1MB  | 27 GB/s | 14 GB/s | **1.86x** |
| 64MB | 39 GB/s | 33 GB/s | **1.18x** |
| 2GB  | 43 GB/s | 37 GB/s | **1.18x** |

See `docs/benchmark/artifacts/` for full benchmark reports.

## Testing

```bash
# Single-process
bazel test //:test_ops_allreduce

# MPI (requires 2 GPUs)
bazel build //:test_ops_allreduce_mpi
CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 --allow-run-as-root bazel-bin/test_ops_allreduce_mpi
```
