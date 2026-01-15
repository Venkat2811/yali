# Yali Examples

Practical examples showing how to use Yali collectives at different levels of abstraction.

## Directory Structure

```
examples/
├── 01_single_process/          # Single process, multiple GPUs
│   └── 01_allreduce/           # AllReduce examples
│       ├── simple.cu           # High-level API
│       └── multilane.cu        # Manual lane configuration
└── 02_multi_process/           # MPI-based examples
    └── 01_allreduce/           # AllReduce examples
        ├── simple_mpi.cu       # High-level MPI API
        └── multilane_mpi.cu    # Manual lane configuration with MPI
```

## Quick Start

```bash
# Build
bazel build //examples/01_single_process/01_allreduce:simple
bazel build //examples/01_single_process/01_allreduce:multilane

# Run (requires 2 GPUs with NVLink)
CUDA_VISIBLE_DEVICES=0,1 bazel-bin/examples/01_single_process/01_allreduce/simple
```

## Example Categories

| #  | Category                               | Description                              |
|:---|:---------------------------------------|:-----------------------------------------|
| 01 | [single_process](01_single_process/)   | Single process controlling multiple GPUs |
| 02 | [multi_process](02_multi_process/)     | MPI-based multi-process examples         |

## API Summary

### Simple API (recommended)

```cpp
#include "src/ops/allreduce.cuh"

yali::Comm comm(0, 1);                                      // Setup
yali::allreduce(comm, send0, recv0, send1, recv1, count);   // AllReduce
```

### Multi-Lane API (full control)

```cpp
#include "yali_launch.h"
#include "src/all_reduce/kernels.cuh"

YaliLaunchArgs args[kLanes];
for (int lane = 0; lane < kLanes; ++lane) {
    args[lane].localInput = send;
    args[lane].localOutput = recv;
    args[lane].peerInput = peer_send;
    args[lane].elementCount = elems_per_lane;
    // ...
}
FlashKernel<float, 3><<<grid, block, smem>>>(args_dev, kLanes, kCtasPerLane);
```

## Performance

All examples achieve the same peak performance as the benchmark harness.

Benchmarked on 2x A100-SXM4-80GB (NVLink NV2, ~47 GB/s unidirectional):

| Message Size | YALI | NCCL | Speedup |
|:-------------|:----:|:----:|:-------:|
| 1 MB         | 27 GB/s | 14 GB/s | **1.85x** |
| 64 MB        | 39 GB/s | 34 GB/s | **1.15x** |
| 2 GB         | 44 GB/s | 37 GB/s | **1.22x** |

Performance is identical across all supported dtypes (FP32, FP16, BF16).

See `docs/benchmark/artifacts/` for full benchmark reports with graphs.
