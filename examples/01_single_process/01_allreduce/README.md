# AllReduce Examples (Single Process)

2-GPU AllReduce examples running in a single process.

## Examples

| File | Description | Complexity |
|------|-------------|------------|
| [simple.cu](simple.cu) | High-level API with auto-tuning | Beginner |
| [multilane.cu](multilane.cu) | Manual lane configuration | Advanced |

## Build & Run

```bash
# Build
bazel build //examples/01_single_process/01_allreduce:simple
bazel build //examples/01_single_process/01_allreduce:multilane

# Run (requires 2 GPUs with NVLink)
CUDA_VISIBLE_DEVICES=0,1 bazel-bin/examples/01_single_process/01_allreduce/simple
CUDA_VISIBLE_DEVICES=0,1 bazel-bin/examples/01_single_process/01_allreduce/multilane
```

## Which Example to Use?

### Start with `simple.cu` if:
- You want the simplest path to AllReduce
- Auto-tuned kernel selection is sufficient
- You don't need fine-grained control

### Use `multilane.cu` if:
- You need custom lane configurations
- You're integrating with existing CUDA infrastructure
- You want to understand the kernel internals

## API Comparison

### Simple API (simple.cu)

```cpp
#include "src/ops/allreduce.cuh"

yali::Comm comm(0, 1);
yali::allreduce(comm, send0, recv0, send1, recv1, count);
```

### Multi-Lane API (multilane.cu)

```cpp
#include "yali_launch.h"
#include "src/all_reduce/kernels.cuh"

YaliLaunchArgs args[kLanes];
// Setup per-lane args...
FlashKernel<float, 3><<<grid, block, smem>>>(args_dev, kLanes, kCtasPerLane);
```

## Performance

Both examples achieve the same peak performance:

| Message Size | Bandwidth | Kernel |
|--------------|-----------|--------|
| 64 MB | ~76 GB/s | Low-latency |
| 2 GB | ~318 GB/s | Bandwidth |
