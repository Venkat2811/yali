# YALI - Setup Guide

This document describes how to set up a reproducible environment for benchmarking YALI AllReduce against NCCL.

See also:
- [README.md](README.md) - Quick start and overview
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Technical deep-dive with diagrams

## Quick Start

```bash
# One-command setup (requires sudo for apt packages)
make setup

# Activate Python environment
source venv-2xa100/bin/activate

# Build everything (auto-detects GPU architecture)
make build-all

# Verify setup with examples
make test-examples

# Run benchmarks
bazel build //:benchmark_yali //:benchmark_nccl
CUDA_VISIBLE_DEVICES=0,1 bazel-bin/benchmark_yali 16777216 20 cuda-events
```

For manual override of GPU architecture:
```bash
make build-all CUDA_ARCH=90  # For H100
make build-all CUDA_ARCH=100 # For B200
```

---

## Prerequisites

### System Requirements

| Requirement | Version                                                    |
|:------------|:-----------------------------------------------------------|
| GPU         | NVIDIA A100/H100/B200 (sm_80/sm_90/sm_100), 2+ with NVLink |
| CUDA        | 12.0+ (tested with CUDA 12.8/13.0)                         |
| OS          | Ubuntu 22.04+ (or similar Linux distribution)              |
| Bazel       | 8.0+ (auto-installed by `make setup`)                      |

### System Packages

**Automated** (recommended):
```bash
make deps  # Installs cmake, bazel, libboost, build-essential
```

**Manual**:
```bash
# Required for building
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libboost-program-options-dev \
    python3 \
    python3-pip \
    python3-venv

# Optional: MPI for multi-process NCCL testing (Mode 3)
sudo apt-get install -y openmpi-bin libopenmpi-dev

# Install Bazel via Bazelisk (auto-downloads correct version)
curl -fsSL https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-linux-amd64 -o /usr/local/bin/bazel
chmod +x /usr/local/bin/bazel

# Verify CUDA installation
nvcc --version
nvidia-smi
```

## Repository Setup

### 1. Clone with Submodules
```bash
git clone --recursive <repo-url>
cd yali

# If already cloned without --recursive:
git submodule update --init --recursive
```

### 2. Verify Submodule Versions
All dependencies are pinned to specific versions for reproducibility:

| Submodule   | Version   | Purpose                      |
|:------------|:----------|:-----------------------------|
| nccl        | v2.28.9-1 | NCCL library (baseline)      |
| nccl-tests  | v2.17.6   | NCCL performance tests       |
| nvbandwidth | v0.8      | NVLink bandwidth measurement |

```bash
# Verify versions
git submodule status
# Expected output:
# dbc86fd... third_party/nccl (v2.28.9-1)
# da0b547... third_party/nccl-tests (v2.17.6)
# 66746a3... third_party/nvbandwidth (v0.8)
```

### 3. Python Environment
```bash
# Create virtual environment (using uv for speed)
uv venv venv-2xa100
source venv-2xa100/bin/activate
uv pip install -r requirements.txt

# Or with standard pip:
python3 -m venv venv-2xa100
source venv-2xa100/bin/activate
pip install -r requirements.txt
```

## Hardware Baseline

Before running benchmarks, capture your hardware baseline to establish ground truth:

```bash
# Quick hardware info (NVLink topology, GPU config)
make hw-info

# Full hardware baseline with nvbandwidth measurements
make hw-baseline
```

This saves system info and bandwidth measurements to `_wip/results/<date>/hw-baseline/`:
- `system_info.txt`: GPU config, NVLink topology, driver versions
- `nvbandwidth_d2d.txt`: D2D bandwidth measurements

### Understanding NVLink Configurations

| Config | NVLinks | Unidirectional | Bidirectional | Typical Systems |
|:-------|:-------:|:--------------:|:-------------:|:----------------|
| NV2    |       2 |      ~47 GB/s  |     ~91 GB/s  | Cloud VMs       |
| NV4    |       4 |      ~95 GB/s  |    ~183 GB/s  | Some DGX        |
| NV6    |       6 |     ~140 GB/s  |    ~274 GB/s  | DGX A100        |
| NV12   |      12 |     ~300 GB/s  |    ~580 GB/s  | DGX B200        |

Your achievable NCCL/Yali bandwidth depends on your NVLink configuration.

## Building via Bazel

All builds are managed via Bazel with `rules_cuda` for proper CUDA compilation.

### Build Benchmarks
```bash
# Build YALI and NCCL benchmarks
bazel build //:benchmark_yali //:benchmark_nccl

# H100 (sm_90)
bazel build //:benchmark_yali --config=h100
```

### Build NCCL + nccl-tests
```bash
bazel build //:nccl_tests_bin
```
**Outputs**: `all_reduce_perf` and `libnccl.so.2`

**Important**: The nccl-tests binary is built against our NCCL v2.28.9 submodule, NOT system NCCL. This is achieved via `-isystem` compiler flag to prioritize our headers.

### Build nvbandwidth
```bash
bazel build //:nvbandwidth_bin
```

### Build MPI Benchmarks
```bash
# Build MPI benchmarks (requires OpenMPI)
bazel build //:benchmark_yali_mpi //:benchmark_nccl_mpi

# Run with 2 ranks (one per GPU)
mpirun --allow-run-as-root -np 2 $(bazel info bazel-bin)/benchmark_yali_mpi 16777216 20 cuda-events
```

### Build Unit Tests
```bash
bazel build //:test_dtypes //:test_validation //:test_peer_access \
    //:test_buffer_ops //:test_all_reduce_correctness //:test_all_reduce_interface
```

### Build Everything
```bash
bazel build //:benchmark_yali //:benchmark_nccl //:nccl_tests_bin //:nvbandwidth_bin
```

### Locate Build Outputs
Bazel outputs are in the execroot. Use `bazel info bazel-bin` to find them:
```bash
BAZEL_BIN=$(bazel info bazel-bin)
ls $BAZEL_BIN/benchmark_yali
ls $BAZEL_BIN/benchmark_nccl
ls $BAZEL_BIN/all_reduce_perf
ls $BAZEL_BIN/nvbandwidth
```

## Validation

### Run Tests
```bash
# Run examples to verify correctness
make test-examples

# Run unit tests
make test-unit
```

### Manual Validation

#### 1. Verify NCCL Version Matching
```bash
BAZEL_BIN=$(bazel info bazel-bin)
CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=$BAZEL_BIN:$LD_LIBRARY_PATH \
    $BAZEL_BIN/all_reduce_perf -g 2 -b 1M -e 1M

# Expected output should show:
# nccl-tests version 2.17.6 nccl-headers=22809 nccl-library=22809
#                                              ^^^^^ MUST MATCH ^^^^^
```

If headers don't match (e.g., `nccl-headers=22602 nccl-library=22809`), the system NCCL headers are being used instead of our submodule. Rebuild with:
```bash
bazel clean
bazel build //:nccl_tests_bin
```

#### 2. Validate nvbandwidth
```bash
BAZEL_BIN=$(bazel info bazel-bin)
CUDA_VISIBLE_DEVICES=0,1 $BAZEL_BIN/nvbandwidth -t device_to_device_memcpy_read_ce

# Output depends on NVLink config (run `make hw-info` to see yours):
# - NV2:  ~47 GB/s unidirectional
# - NV4:  ~95 GB/s unidirectional
# - NV6:  ~140 GB/s unidirectional
# - NV12: ~300 GB/s unidirectional
```

#### 3. Validate Yali
```bash
BAZEL_BIN=$(bazel info bazel-bin)
CUDA_VISIBLE_DEVICES=0,1 $BAZEL_BIN/benchmark_yali 16777216 20 cuda-events

# Expected: ~75-80 GB/s at 64MB
```

#### 4. Run Examples
```bash
BAZEL_BIN=$(bazel info bazel-bin)
bazel build //:example_simple
CUDA_VISIBLE_DEVICES=0,1 $BAZEL_BIN/example_simple

# Expected: GPU0[0]=3.0, GPU1[0]=3.0 (correctness verification)
```

#### 5. Validate NCCL Baseline
```bash
BAZEL_BIN=$(bazel info bazel-bin)
CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=$BAZEL_BIN:$LD_LIBRARY_PATH \
    $BAZEL_BIN/all_reduce_perf -g 2 -b 128M -e 128M -w 1 -n 5

# Bus bandwidth depends on NVLink config:
# - NV2:  ~36-39 GB/s (75-83% of 47 GB/s peak)
# - NV4:  ~65-73 GB/s (~75% of 93 GB/s peak)
# - NV6:  ~90-96 GB/s (~70% of 140 GB/s peak)
# - NV12: ~240+ GB/s
```

## Benchmark Sweeps

### Single Command Full Sweep (Recommended)

```bash
# Complete sweep: hw-baseline + examples + yali + nccl + auto-generated report
make sweep              # Full sweep (all dtypes: FP32, FP16, BF16)
make sweep-mpi          # MPI mode (all dtypes)
make sweep-quick        # Quick: FP32 only, both single-process AND MPI
```

Results saved to `output/YYYY-MM-DD/HHMMSS/`:

| Directory                  | Contents                                  |
|:---------------------------|:------------------------------------------|
| `hw-baseline/`             | nvbandwidth D2D measurements, system info |
| `examples/`                | Example correctness results               |
| `yali/fp32.csv`            | YALI sweep (mean, stddev, min, max)       |
| `yali/fp16.csv`, `bf16.csv`| Per-dtype results                         |
| `nccl/fp32.csv`, etc.      | NCCL baseline per dtype                   |
| `summary.md`               | Auto-generated comparison report          |

The report includes:
- Hardware baseline (nvbandwidth D2D)
- Executive summary with peak bandwidth and SoL %
- Detailed per-dtype comparison tables
- YALI vs NCCL speedup calculations

### Quick Benchmark

```bash
# Fast comparison (5 sizes, single run)
make bench              # Single-process
make bench-mpi          # MPI mode
```

## NCCL Execution Modes

NCCL supports three execution modes for multi-GPU communication. We provide Makefile targets for all three:

### Mode 1: Single-Process, 1 Thread, 2 GPUs (`-g 2`)
```bash
make sweep-nccl-1proc-1thr
```
One process manages both GPUs. Simplest setup, no MPI required.

### Mode 2: Single-Process, 2 Threads, 1 GPU each (`-t 2 -g 1`)
```bash
make sweep-nccl-1proc-2thr
```
One process with pthreads, each thread managing one GPU.

### Mode 3: Multi-Process via MPI (`mpirun -np 2 -g 1`)
```bash
# First, build NCCL with MPI support
make setup-mpi      # Installs OpenMPI if needed
make build-nccl-mpi # Builds NCCL and nccl-tests with MPI

# Then run the sweep
make sweep-nccl-2proc-mpi
```
Two separate processes (via MPI), each managing one GPU. Most realistic for distributed training.

### Run All Modes
```bash
make sweep-nccl-all-modes  # Runs all 3 modes sequentially
```

Results are saved to `output/<date>/nccl-<mode>/`.

## Architecture Notes

### SM Architecture
Default builds target **sm_80** (A100). Use `--config` for different GPUs:
```bash
bazel build //:benchmark_yali --config=a100  # A100 (default)
bazel build //:benchmark_yali --config=h100  # H100
```

### NVLink Bandwidth Reference
Bandwidth varies by NVLink configuration (use `make hw-baseline` to measure yours):

| Config | Unidirectional | Bidirectional | Typical Systems  |
|:-------|:--------------:|:-------------:|:-----------------|
| NV2    |      ~47 GB/s  |     ~91 GB/s  | Cloud VMs        |
| NV4    |      ~95 GB/s  |    ~183 GB/s  | Some DGX systems |
| NV6    |     ~140 GB/s  |    ~274 GB/s  | DGX A100         |
| NV12   |     ~300 GB/s  |    ~580 GB/s  | DGX B200         |

### SoL (Speed of Light) Calculation
```
SoL% = (measured_bus_bw / nvlink_peak) × 100
```
Where `nvlink_peak` is measured via nvbandwidth `device_to_device_memcpy_read_ce`.

**Note**: Yali can exceed 100% SoL because it uses bidirectional NVLink, while the SoL reference is unidirectional.

### Timing Modes

The benchmark supports three timing modes:

| Mode | Argument | Description | Use Case |
|:-----|:---------|:------------|:---------|
| **throughput** | `throughput` | Wall-clock, fire-and-forget | Production-like |
| **latency** | `latency` | Wall-clock, sync after each call | BS=1 interactive |
| **cuda-events** | `cuda-events` | GPU-only timing | Fair comparison with NCCL |

```bash
# CUDA events timing (recommended for benchmarking)
CUDA_VISIBLE_DEVICES=0,1 $BAZEL_BIN/benchmark_yali 16777216 20 cuda-events

# Throughput timing (production-like)
CUDA_VISIBLE_DEVICES=0,1 $BAZEL_BIN/benchmark_yali 16777216 20 throughput
```

For detailed kernel profiling, use Nsight Systems:
```bash
nsys profile --stats=true -o yali_profile $BAZEL_BIN/benchmark_yali 16777216 20 throughput
```

## Troubleshooting

### Bazel not found
Install Bazelisk (auto-downloads correct Bazel version):
```bash
sudo curl -fsSL https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-linux-amd64 -o /usr/local/bin/bazel
sudo chmod +x /usr/local/bin/bazel
```
Or run `make deps` to install automatically.

### "Out of bounds values: FAILED" in nccl-tests
This usually indicates NCCL header/library version mismatch. Check:
```bash
# Verify which libnccl.so is being loaded
BAZEL_BIN=$(bazel info bazel-bin)
LD_LIBRARY_PATH=$BAZEL_BIN:$LD_LIBRARY_PATH ldd $BAZEL_BIN/all_reduce_perf | grep nccl

# Should show: libnccl.so.2 => /path/to/bazel-bin/libnccl.so.2
# NOT: /usr/lib/x86_64-linux-gnu/libnccl.so.2
```

### Build fails with "nccl.h not found"
Ensure NCCL submodule is initialized:
```bash
git submodule update --init third_party/nccl
```

### nvbandwidth cmake errors
Install required dependencies:
```bash
sudo apt-get install cmake libboost-program-options-dev
```

### GPU not detected
```bash
# Check GPU visibility
nvidia-smi -L

# Set specific GPUs
export CUDA_VISIBLE_DEVICES=0,1
```

### tabulate not found for report generation
Activate the Python virtual environment:
```bash
source venv-2xa100/bin/activate
```

## File Structure

```
yali/
├── Makefile                     # Automated setup/build/test (run `make help`)
├── MODULE.bazel                 # Bazel bzlmod config (rules_cuda)
├── BUILD.bazel                  # Bazel build rules
├── .bazelrc                     # Bazel configuration (arch flags)
├── .bazelversion                # Bazel version (8.0.0)
├── README.md                    # Project overview
├── SETUP.md                     # This file
├── requirements.txt             # Python dependencies (pynvml, tabulate)
│
├── bench/
│   ├── benchmark_yali.cu        # YALI benchmark
│   ├── benchmark_nccl.cu        # NCCL benchmark
│   ├── benchmark_yali_mpi.cu    # YALI MPI benchmark
│   └── benchmark_nccl_mpi.cu    # NCCL MPI benchmark
│
├── src/
│   ├── include/                 # PUBLIC headers (users include from here)
│   │   ├── yali.h               # Single public header (like nccl.h)
│   │   ├── yali_launch.h        # Launch arguments struct (advanced)
│   │   └── yali_tuning.h        # Auto-tuning heuristics
│   ├── kernels/                 # CUDA kernel implementations
│   │   ├── stream.cu            # Stream kernel (large messages)
│   │   ├── stream.cuh           # Stream kernel header
│   │   ├── ring.cuh             # Ring buffer primitives
│   │   └── type_ops.cuh         # Dtype-specific operations
│   ├── all_reduce/              # AllReduce interface
│   │   ├── all_reduce.h         # Internal API
│   │   ├── kernels.cuh          # Kernel dispatch
│   │   ├── stream.cuh           # Stream kernel wrapper
│   │   └── flash.cuh            # Flash kernel wrapper
│   ├── ops/                     # High-level ops API
│   │   └── allreduce.cuh        # User-facing AllReduce API
│   └── common/                  # Shared utilities
│       ├── buffer_ops.cuh       # Buffer initialization/validation
│       ├── hw_info.cuh          # Hardware detection
│       ├── peer_access.cuh      # GPU peer access setup
│       └── validation.cuh       # Result validation
│
├── scripts/
│   ├── sweep.py                 # Comprehensive benchmark sweep
│   └── quick_benchmark.py       # Fast YALI vs NCCL comparison
│
├── examples/
│   ├── 01_single_process/
│   │   └── 01_allreduce/
│   │       ├── simple.cu        # High-level ops API example
│   │       └── multilane.cu     # Manual lane configuration
│   └── 02_multi_process/
│       └── 01_allreduce/
│           ├── simple_mpi.cu    # MPI high-level API
│           └── multilane_mpi.cu # MPI manual lane configuration
│
├── tests/
│   └── unit/                    # C++ unit tests
│
├── third_party/
│   ├── nccl/                    # NCCL submodule (v2.28.9-1)
│   ├── nccl-tests/              # nccl-tests submodule (v2.17.6)
│   └── nvbandwidth/             # nvbandwidth submodule (v0.8)
│
└── output/                      # Benchmark results (timestamped)
    └── YYYY-MM-DD/HHMMSS/       # Per-sweep results
        ├── hw-baseline/         # nvbandwidth measurements
        ├── examples/            # Example correctness tests
        ├── yali/                # YALI sweep CSVs (fp32.csv, etc.)
        ├── nccl/                # NCCL sweep results
        └── summary.md           # Auto-generated report
```
