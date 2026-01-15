<p align="center">
  <img src="assets/yali-banner.png" alt="YALI" width="600">
</p>

# YALI - Yet Another Low-Latency Implementation

**2.2x faster than NCCL at 1MB. 106x more stable tail latency at 256MB.**

YALI is a 2-GPU NVLink AllReduce library that outperforms NVIDIA NCCL across the entire message size range (1.1x-2.2x), with profiler-verified benchmarks using NCCL's own busBw convention.

This is not a wrapper around NCCL. YALI is a ground-up implementation, starting with AllReduce and expanding to a full collective API.

Built applying high-performance computing principles proven in HFT systems, distributed databases, and lock-free data structures: **static scheduling**, **prefetching**, and **pre-allocation**. Hardware likes predictability. YALI delivers it.

Two kernel modes, one goal:
- **Flash** - 3-stage double-buffered cp.async prefetch for latency-sensitive workloads (≤64MB)
- **Stream** - 128-lane ring buffer for bandwidth saturation (>64MB)

---

The name comes from **Yali** (யாழி / யாளி) - a composite creature from Tamil and South Indian temple architecture, depicted as part lion, part elephant, part serpent. Like the sphinx or griffin in other cultures, it represents a guardian figure.

*YALI - Yet Another Low-Latency Implementation* - guarding your GPU efficiency.

---

## Key Features

- **Simple API**: 3 lines of code for AllReduce (see below)
- **Two kernel modes**: Flash (small messages) and Stream (large messages)
- **Dtype support**: FP32, FP16, BF16
- **Single & Multi-process**: Both single-process and MPI multi-process support
- **1.1x-2.2x faster than NCCL** across all sizes
- **87% Speed-of-Light**: Near-optimal NVLink utilization
- **106x more stable**: Dramatically lower tail latency variance

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for technical details on how YALI achieves this performance.

## Simple API Usage

```cpp
#include "src/ops/allreduce.cuh"

// Setup (once)
yali::Comm comm(0, 1);  // GPU 0 and 1

// AllReduce: reads from send buffers, writes sum to recv buffers
yali::allreduce(comm, send0, recv0, send1, recv1, count);
```

See `examples/01_single_process/01_allreduce/simple.cu` for a complete working example.

*Built in collaboration with [Claude Code](https://claude.ai/code) and [Codex CLI](https://github.com/openai/codex)*

---

## Quick Start

```bash
# 1. Clone and setup (one-time)
git clone --recursive <repo-url>
cd yali
make setup && source venv-2xa100/bin/activate

# 2. Build (includes YALI + NCCL benchmarks)
make build-all

# 3. Quick benchmark: YALI vs NCCL comparison
python scripts/quick_benchmark.py                    # Single-process mode
python scripts/quick_benchmark.py --mpi              # MPI mode (2 processes)
python scripts/quick_benchmark.py --sizes 64M 128M   # Custom sizes
```

### Sample Output (2x A100 NV4)
```
+-------+------------+-------+------------+-------+---------+
| Dtype | YALI Peak  | SoL % | NCCL Peak  | SoL % | Speedup |
+-------+------------+-------+------------+-------+---------+
| FP32  | 81.95 GB/s | 87.4% | 72.42 GB/s | 77.3% |  1.13x  |
| FP16  | 82.14 GB/s | 87.6% | 69.04 GB/s | 73.7% |  1.19x  |
| BF16  | 82.26 GB/s | 87.8% | 72.60 GB/s | 77.5% |  1.13x  |
+-------+------------+-------+------------+-------+---------+

FP32 Detailed (CUDA Events timing):
+--------+-------------+-------------+---------+
|   Size | YALI (GB/s) | NCCL (GB/s) | Speedup |
+--------+-------------+-------------+---------+
|   1 MB |    39.9     |    17.9     |  2.23x  |
|  64 MB |    76.2     |    63.2     |  1.21x  |
| 128 MB |    79.2     |    67.1     |  1.18x  |
|   2 GB |    81.9     |    72.4     |  1.13x  |
+--------+-------------+-------------+---------+
```

### Manual Benchmark Commands
```bash
# Single benchmark run
CUDA_VISIBLE_DEVICES=0,1 bazel-bin/benchmark_yali 16777216 20 cuda-events  # 64MB
CUDA_VISIBLE_DEVICES=0,1 bazel-bin/benchmark_nccl 16777216 20 cuda-events

# Run examples
CUDA_VISIBLE_DEVICES=0,1 bazel-bin/example_simple
```

## Requirements

- CUDA 12.0+ (tested with CUDA 12.8/13.0)
- 2x NVIDIA GPUs with NVLink (A100, H100, B200)
- Bazel 8.0+ (auto-installed by `make setup`)
- Python 3.8+ with `uv` or `pip`

## Build

```bash
# Build everything (auto-detects GPU architecture)
make build-all

# Or build individually
bazel build //:benchmark_yali   # YALI benchmark
bazel build //:benchmark_nccl   # NCCL benchmark
bazel build //:example_simple   # Simple example

# Build with specific GPU architecture
bazel build //:benchmark_yali --config=h100  # H100
```

## Benchmark Sweeps

### Full Sweep (Recommended)

```bash
# Comprehensive sweep: system info + nvbandwidth + examples + YALI + NCCL
make sweep              # Full sweep (all dtypes: FP32, FP16, BF16)
make sweep-mpi          # MPI mode (all dtypes)
make sweep-quick        # Quick: FP32 only, both single-process AND MPI

# Quick comparison (5 sizes, fast)
make bench              # Quick YALI vs NCCL comparison
make bench-mpi          # MPI mode
```

Output saved to `output/YYYY-MM-DD/HHMMSS/`:
- `hw-baseline/` - System info, nvbandwidth measurements
- `examples/` - Example correctness results
- `yali/fp32.csv`, `yali/fp16.csv`, `yali/bf16.csv` - Per-dtype results
- `nccl/fp32.csv`, etc. - NCCL baseline
- `summary.md` - Auto-generated comparison report with tables

### Sweep Options

```bash
# Direct Python usage for more control
python scripts/sweep.py --quick              # Quick mode (FP32 only)
python scripts/sweep.py --runs 5             # 5 runs per size (more statistics)
python scripts/sweep.py --sizes 16M 64M 2G   # Custom sizes
python scripts/sweep.py --mpi                # MPI mode
```

### NCCL Execution Modes

```bash
# NCCL sweeps (3 execution modes)
make sweep-nccl-1proc-1thr   # Mode 1: -g 2 (single process, 2 GPUs)
make sweep-nccl-1proc-2thr   # Mode 2: -t 2 -g 1 (threaded)
make sweep-nccl-2proc-mpi    # Mode 3: mpirun -np 2 (MPI)
```

## Hardware Baseline

```bash
make hw-info      # Quick GPU/NVLink config summary
make hw-baseline  # Full nvbandwidth measurements
```

## Performance Results (2x A100-SXM4-80GB, NV4)

Benchmarked with CUDA events timing on 2x A100-SXM4-80GB with NV4 (4 NVLinks @ 25 GB/s each = 93.7 GB/s unidirectional):

### Single-Process (2 GPUs, FP32)

| Size   | YALI (GB/s) | NCCL (GB/s) | Speedup | SoL % |
|:-------|:-----------:|:-----------:|:-------:|:-----:|
| 1 MB   |    39.9     |    17.9     | **2.23x** | 43%   |
| 4 MB   |    59.8     |    40.2     | **1.49x** | 64%   |
| 16 MB  |    70.6     |    55.1     | **1.28x** | 75%   |
| 64 MB  |    76.2     |    63.2     | **1.21x** | 81%   |
| 128 MB |    79.2     |    67.1     | **1.18x** | 85%   |
| 2 GB   |    81.9     |    72.4     | **1.13x** | 87%   |

**Key insights:**
- **YALI wins at ALL sizes** with 1.13-2.23x speedup
- **Peak 87% SoL** (81.9 GB/s vs 93.7 GB/s theoretical)
- **2x faster at small sizes** (1-4MB) where latency dominates
- NCCL caps at ~77% SoL due to ring algorithm's unidirectional NVLink usage

## Environment Variables

### Production (user-facing)

| Variable               | Default  | Description                                     |
|:-----------------------|:---------|:------------------------------------------------|
| `CUDA_VISIBLE_DEVICES` | `0,1`    | GPU indices                                     |
| `YALI_ELEMS`           | 33554432 | Elements per rank                               |
| `YALI_DTYPE`           | `fp32`   | Data type (`fp32`, `fp16`, `bf16`)              |
| `YALI_KERNEL_MODE`     | `auto`   | Kernel selection: `auto`, `flash`, `stream`     |
| `YALI_DEBUG`           | 0        | Enable debug output                             |
| `YALI_CUDA_EVENTS`     | 0        | Use CUDA events timing (1) vs wall-clock (0)    |

### Dev/Tuning (prefix `YALI_DEV_`)

| Variable                 | Default | Description                        |
|:-------------------------|:--------|:-----------------------------------|
| `YALI_DEV_LANES`         | auto    | Manual lane count override (1-128) |
| `YALI_DEV_SLOT_BYTES`    | auto    | Ring buffer slot size              |
| `YALI_DEV_CTAS_PER_LANE` | auto    | CTAs per lane (flash kernel)       |
| `YALI_DEV_WARMUP`        | 1       | Warmup iterations                  |
| `YALI_DEV_ITERS`         | 5       | Measurement iterations             |

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed technical documentation with ASCII diagrams.

### Two Kernel Modes

**Flash kernel** (≤64MB messages):
- Direct GPU-to-GPU peer access via `cp.async`
- 3-stage prefetch pipeline hides memory latency
- Multi-CTA parallelism per lane
- ~76 GB/s (81% SoL)

**Stream kernel** (>64MB messages):
- Ring buffer with sequence-based flow control
- Bidirectional NVLink utilization
- Fire-and-forget kernel launches
- ~82 GB/s (87% SoL)

### Key Directories

| Directory         | Purpose                                        |
|:------------------|:-----------------------------------------------|
| `src/include/`    | Public headers (yali.h, yali_launch.h)         |
| `src/kernels/`    | CUDA kernels (stream, flash, ring buffer)      |
| `src/ops/`        | High-level ops API (allreduce.cuh)             |
| `src/all_reduce/` | AllReduce interface and kernel headers         |
| `bench/`          | Benchmarks (benchmark_yali.cu, benchmark_nccl.cu) |
| `examples/`       | Example code (simple, multilane)               |
| `scripts/`        | Python utilities (sweep.py, quick_benchmark.py)|
| `third_party/`    | Submodules (nccl, nccl-tests, nvbandwidth)     |

See [SETUP.md](SETUP.md) for the complete directory structure.

## Submodules

| Submodule   | Version   | Purpose                           |
|:------------|:----------|:----------------------------------|
| nccl        | v2.28.9-1 | NCCL library (baseline + headers) |
| nccl-tests  | v2.17.6   | NCCL performance tests            |
| nvbandwidth | v0.8      | NVLink bandwidth measurement      |

Initialize:
```bash
git submodule update --init --recursive
```

## Validation

```bash
# Run examples to verify correctness
make test-examples

# Run unit tests
make test-unit
```

## Limitations

- **2 GPUs only**: Hardcoded for 2-GPU configurations
- **NVLink required**: Requires direct GPU-to-GPU peer access
- **Single-node**: No multi-node support (single-node MPI supported)

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Technical deep-dive with ASCII diagrams
- [SETUP.md](SETUP.md) - Detailed setup and configuration guide
- `output/` - Benchmark results (gitignored)

## License

See LICENSE file.
