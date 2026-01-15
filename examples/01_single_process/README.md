# Single Process Examples

Examples that run in a single process controlling multiple GPUs.

## Operations

| #  | Name                         | Description                      |
|:---|:-----------------------------|:---------------------------------|
| 01 | [allreduce](01_allreduce/)   | Sum across all GPUs, result on all |

## Requirements

- 2+ GPUs with NVLink/NVSwitch
- P2P access enabled between GPUs

## Usage Pattern

All single-process examples follow this pattern:

```cpp
// 1. Setup communicator
yali::Comm comm(gpu0, gpu1);

// 2. Allocate buffers on each GPU
cudaSetDevice(gpu0);
cudaMalloc(&buf0, size);
cudaSetDevice(gpu1);
cudaMalloc(&buf1, size);

// 3. Call collective
yali::allreduce(comm, ...);
```
