# Makefile for YALI benchmarks
#
# Quick start:
#   make setup && source venv-2xa100/bin/activate && make build-all && make validate

.PHONY: all setup deps submodules venv build build-all build-yali build-nccl build-nvbandwidth \
        build-yali-mpi build-nccl-mpi build-examples build-examples-mpi build-unit-tests setup-mpi \
        test test-unit test-unit-cpp test-all test-examples test-examples-mpi test-ops test-ops-mpi \
        test-correctness test-perf test-mpi-all test-mpi-correctness validate \
        sweep sweep-quick sweep-standard sweep-extensive sweep-single-only sweep-mpi-only bench bench-mpi \
        sweep-nccl-1proc-1thr sweep-nccl-1proc-2thr sweep-nccl-2proc-mpi sweep-nccl-all-modes \
        hw-baseline hw-info info run-yali run-nccl run-nvbandwidth run-yali-mpi \
        clean clean-all help detect-arch \
        format format-cpp format-bazel lint lint-cpp lint-bazel setup-formatters

# GPU Architecture: auto-detect or manual override
# Usage: make build-all CUDA_ARCH=90  (manual override for H100)
# Default: auto-detect from nvidia-smi
CUDA_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifeq ($(CUDA_ARCH),)
    CUDA_ARCH := 80
endif

# Results directory for benchmark outputs
RESULTS_DIR ?= output/$(shell date +%Y-%m-%d)

# Default target
all: setup build-all validate

# === Setup Targets ===

setup: deps submodules venv
	@echo ""
	@echo "=== Setup Complete ==="
	@echo "Detected GPU architecture: sm_$(CUDA_ARCH)"
	@echo ""
	@echo "Next steps:"
	@echo "  source venv-2xa100/bin/activate"
	@echo "  make build-all"
	@echo "  make validate"

deps:
	@echo "=== Installing System Dependencies ==="
	@command -v cmake >/dev/null 2>&1 || (echo "Installing cmake..." && sudo apt-get update -qq && sudo apt-get install -y -qq cmake)
	@command -v bazel >/dev/null 2>&1 || (echo "Installing bazelisk..." && sudo curl -fsSL https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-linux-amd64 -o /usr/local/bin/bazel && sudo chmod +x /usr/local/bin/bazel)
	@dpkg -s libboost-program-options-dev >/dev/null 2>&1 || (echo "Installing libboost-program-options-dev..." && sudo apt-get install -y -qq libboost-program-options-dev)
	@dpkg -s build-essential >/dev/null 2>&1 || (echo "Installing build-essential..." && sudo apt-get install -y -qq build-essential)
	@echo "System dependencies OK"

# Install MPI for multi-process NCCL testing (Mode 3)
setup-mpi:
	@echo "=== Installing OpenMPI ==="
	@if command -v mpirun >/dev/null 2>&1; then \
		echo "MPI already installed: $$(mpirun --version | head -1)"; \
	else \
		echo "Installing openmpi-bin and libopenmpi-dev..."; \
		sudo apt-get install -y openmpi-bin libopenmpi-dev; \
		echo "MPI installed: $$(mpirun --version | head -1)"; \
	fi

submodules:
	@echo "=== Initializing Git Submodules ==="
	git submodule update --init --recursive
	@echo "Submodule versions:"
	@git submodule status

venv:
	@echo "=== Creating Python Virtual Environment ==="
	@if command -v uv >/dev/null 2>&1; then \
		echo "Using uv..."; \
		uv venv venv-2xa100; \
		. venv-2xa100/bin/activate && uv pip install -r requirements.txt; \
	else \
		echo "Using pip..."; \
		python3 -m venv venv-2xa100; \
		. venv-2xa100/bin/activate && pip install -q -r requirements.txt; \
	fi
	@echo "Python environment ready"

# === GPU Architecture Detection ===

detect-arch:
	@echo "Detected GPU architecture: sm_$(CUDA_ARCH)"
	@nvidia-smi --query-gpu=name,compute_cap --format=csv

# === Build Targets ===

build: build-yali

# Build YALI benchmark
build-yali:
	@echo "Building YALI benchmark for sm_$(CUDA_ARCH)..."
	bazel build //:benchmark_yali //:benchmark_nccl

build-nccl:
	@echo "Building NCCL + nccl-tests for sm_$(CUDA_ARCH)..."
	bazel build //:nccl_tests_bin

build-nvbandwidth:
	@echo "Building nvbandwidth for sm_$(CUDA_ARCH)..."
	bazel build //:nvbandwidth_bin

build-nccl-mpi:
	@echo "Building NCCL + nccl-tests with MPI support for sm_$(CUDA_ARCH)..."
	@if ! command -v mpicc >/dev/null 2>&1; then \
		echo "ERROR: MPI not found. Run: make setup-mpi"; \
		exit 1; \
	fi
	@# Build NCCL first if not already built
	@if [ ! -f "nccl/build/lib/libnccl.so" ]; then \
		echo "Building NCCL library..."; \
		$(MAKE) -C nccl -j$$(nproc) src.build NVCC_GENCODE="-gencode=arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)"; \
	fi
	@# Build nccl-tests with MPI
	@echo "Building nccl-tests with MPI support..."
	@MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi && \
	rm -rf nccl-tests/build && \
	$(MAKE) -C nccl-tests -j$$(nproc) \
		MPI=1 \
		MPI_HOME=$$MPI_HOME \
		CUDA_HOME=$${CUDA_HOME:-/usr/local/cuda} \
		NCCL_HOME=$$(pwd)/nccl/build \
		NVCC_GENCODE="-gencode=arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)"
	@echo "MPI-enabled nccl-tests built: nccl-tests/build/all_reduce_perf"

# Build YALI MPI benchmarks
build-yali-mpi:
	@echo "Building YALI MPI benchmarks for sm_$(CUDA_ARCH)..."
	@if ! command -v mpicc >/dev/null 2>&1; then \
		echo "ERROR: MPI not found. Run: make setup-mpi"; \
		exit 1; \
	fi
	bazel build //:benchmark_yali_mpi //:benchmark_nccl_mpi

build-all: build-yali build-nccl build-nvbandwidth build-unit-tests
	@echo ""
	@echo "=== Build Complete ==="
	@echo "BAZEL_BIN=$$(bazel info bazel-bin)"
	@echo "GPU Architecture: sm_$(CUDA_ARCH)"

# === Test Targets ===

test: test-correctness

test-correctness:
	bazel test //:correctness_tests --test_output=summary

test-perf:
	bazel test //:benchmarks --test_output=all

validate:
	@echo "=== Running Validation Suite ==="
	bazel test //:validation --test_output=all

# === Unit Test Targets ===

# Build C++ unit tests using cuda_library (incremental builds)
build-unit-tests:
	@echo "=== Building C++ Unit Tests for sm_$(CUDA_ARCH) ==="
	bazel build //:test_dtypes //:test_all_reduce_correctness //:test_validation \
		//:test_peer_access //:test_buffer_ops //:test_all_reduce_interface

# Run C++ unit tests
test-unit-cpp: build-unit-tests
	@echo "=== Running C++ Unit Tests ==="
	@BAZEL_BIN=$$(bazel info bazel-bin) && \
	CUDA_VISIBLE_DEVICES=0,1 $$BAZEL_BIN/test_dtypes && \
	CUDA_VISIBLE_DEVICES=0,1 $$BAZEL_BIN/test_all_reduce_correctness && \
	CUDA_VISIBLE_DEVICES=0,1 $$BAZEL_BIN/test_validation && \
	CUDA_VISIBLE_DEVICES=0,1 $$BAZEL_BIN/test_peer_access && \
	CUDA_VISIBLE_DEVICES=0,1 $$BAZEL_BIN/test_buffer_ops && \
	CUDA_VISIBLE_DEVICES=0,1 $$BAZEL_BIN/test_all_reduce_interface

# Run shell-based integration tests (exercises full harness)
test-unit: build-yali
	@echo "=== Running Integration Unit Tests ==="
	@./tests/unit/run_unit_tests.sh

test-all: test-unit-cpp test-unit test-correctness test-examples test-ops
	@echo "=== All Tests Passed ==="

# === Example and Ops API Tests ===

# Build examples
build-examples:
	@echo "=== Building Examples ==="
	bazel build //:example_simple //:example_multilane //:test_ops_allreduce

# Build MPI examples
build-examples-mpi:
	@echo "=== Building MPI Examples ==="
	@if ! command -v mpicc >/dev/null 2>&1; then \
		echo "ERROR: MPI not found. Run: make setup-mpi"; \
		exit 1; \
	fi
	bazel build //:example_simple_mpi //:example_multilane_mpi

# Test examples (correctness)
test-examples: build-examples
	@echo "=== Testing Examples ==="
	@BAZEL_BIN=$$(bazel info bazel-bin) && \
	echo "--- simple.cu ---" && \
	CUDA_VISIBLE_DEVICES=0,1 $$BAZEL_BIN/example_simple && \
	echo "--- multilane.cu ---" && \
	CUDA_VISIBLE_DEVICES=0,1 $$BAZEL_BIN/example_multilane && \
	echo "=== All Examples Passed ==="

# Test MPI examples (correctness)
test-examples-mpi: build-examples-mpi
	@echo "=== Testing MPI Examples ==="
	@BAZEL_BIN=$$(bazel info bazel-bin) && \
	echo "--- simple_mpi.cu ---" && \
	CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 --allow-run-as-root --bind-to none \
		-x CUDA_VISIBLE_DEVICES $$BAZEL_BIN/example_simple_mpi && \
	echo "--- multilane_mpi.cu ---" && \
	CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 --allow-run-as-root --bind-to none \
		-x CUDA_VISIBLE_DEVICES $$BAZEL_BIN/example_multilane_mpi && \
	echo "=== All MPI Examples Passed ==="

# Test ops API (correctness + performance)
test-ops: build-examples
	@echo "=== Testing Ops API ==="
	@BAZEL_BIN=$$(bazel info bazel-bin) && \
	CUDA_VISIBLE_DEVICES=0,1 $$BAZEL_BIN/test_ops_allreduce

# Test ops API MPI (correctness + performance)
test-ops-mpi: build-yali-mpi
	@echo "=== Testing Ops API (MPI) ==="
	@BAZEL_BIN=$$(bazel info bazel-bin) && \
	CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 --allow-run-as-root --bind-to none \
		-x CUDA_VISIBLE_DEVICES $$BAZEL_BIN/test_ops_allreduce_mpi

# Test MPI correctness
test-mpi-correctness: build-yali-mpi
	@echo "=== Running MPI Correctness Tests ==="
	@./tests/run_yali_mpi_correctness.sh

# Test all MPI
test-mpi-all: test-mpi-correctness test-examples-mpi
	@echo "=== All MPI Tests Passed ==="

# === Quick Run Targets ===

# Run YALI benchmark (64MB, 20 calls, cuda-events timing)
run-yali:
	@BAZEL_BIN=$$(bazel info bazel-bin) && \
	CUDA_VISIBLE_DEVICES=0,1 $$BAZEL_BIN/benchmark_yali 16777216 20 cuda-events

run-nccl:
	@BAZEL_BIN=$$(bazel info bazel-bin) && \
	CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH="$$BAZEL_BIN" \
	$$BAZEL_BIN/all_reduce_perf -g 2 -b 128M -e 128M -w 1 -n 5

run-nvbandwidth:
	@BAZEL_BIN=$$(bazel info bazel-bin) && \
	CUDA_VISIBLE_DEVICES=0,1 $$BAZEL_BIN/nvbandwidth -t device_to_device_memcpy_read_ce

# Run YALI MPI benchmark
run-yali-mpi: build-yali-mpi
	@BAZEL_BIN=$$(bazel info bazel-bin) && \
	CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 --allow-run-as-root --bind-to none \
		-x CUDA_VISIBLE_DEVICES \
		$$BAZEL_BIN/benchmark_yali_mpi 16777216 20 cuda-events

# === Utility Targets ===

clean:
	bazel clean
	rm -rf venv-2xa100

clean-all: clean
	rm -rf nccl/build nccl-tests/build

info:
	@echo "=== Environment Info ==="
	@echo "BAZEL_BIN=$$(bazel info bazel-bin 2>/dev/null || echo 'not built')"
	@echo "CUDA_ARCH=$(CUDA_ARCH)"
	@echo ""
	@echo "=== GPU Info ==="
	@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
	@echo ""
	@echo "=== Submodule Status ==="
	@git submodule status

# === Format/Lint Targets ===

# Install formatters (clang-format, buildifier)
setup-formatters:
	@echo "=== Installing Formatters ==="
	@if ! command -v clang-format >/dev/null 2>&1; then \
		echo "Installing clang-format..."; \
		sudo apt-get update -qq && sudo apt-get install -y -qq clang-format; \
	else \
		echo "clang-format: $$(clang-format --version | head -1)"; \
	fi
	@if ! command -v buildifier >/dev/null 2>&1; then \
		echo "Installing buildifier..."; \
		sudo curl -fsSL https://github.com/bazelbuild/buildtools/releases/download/v7.3.1/buildifier-linux-amd64 -o /usr/local/bin/buildifier; \
		sudo chmod +x /usr/local/bin/buildifier; \
	else \
		echo "buildifier: $$(buildifier --version)"; \
	fi
	@echo "Formatters ready"

# Source file globs for formatting (exclude submodules)
CPP_SOURCES := $(shell find harness device include src tests/unit -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.cpp' 2>/dev/null)
BAZEL_FILES := BUILD.bazel MODULE.bazel

# Format all code
format: format-cpp format-bazel
	@echo "=== All files formatted ==="

# Format C++/CUDA code with clang-format
format-cpp:
	@if ! command -v clang-format >/dev/null 2>&1; then \
		echo "ERROR: clang-format not found. Run: make setup-formatters"; \
		exit 1; \
	fi
	@echo "=== Formatting C++/CUDA files ==="
	@find harness device include src tests/unit -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.cpp' 2>/dev/null | \
		xargs -r clang-format -i
	@echo "C++/CUDA files formatted"

# Format Bazel files with buildifier
format-bazel:
	@if ! command -v buildifier >/dev/null 2>&1; then \
		echo "ERROR: buildifier not found. Run: make setup-formatters"; \
		exit 1; \
	fi
	@echo "=== Formatting Bazel files ==="
	@buildifier BUILD.bazel MODULE.bazel
	@echo "Bazel files formatted"

# Lint all code (check without modifying)
lint: lint-cpp lint-bazel
	@echo "=== All lint checks passed ==="

# Lint C++/CUDA code (check only)
lint-cpp:
	@if ! command -v clang-format >/dev/null 2>&1; then \
		echo "ERROR: clang-format not found. Run: make setup-formatters"; \
		exit 1; \
	fi
	@echo "=== Checking C++/CUDA formatting ==="
	@find harness device include src tests/unit -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.cpp' 2>/dev/null | \
		xargs -r clang-format --dry-run --Werror
	@echo "C++/CUDA formatting OK"

# Lint Bazel files (check only)
lint-bazel:
	@if ! command -v buildifier >/dev/null 2>&1; then \
		echo "ERROR: buildifier not found. Run: make setup-formatters"; \
		exit 1; \
	fi
	@echo "=== Checking Bazel formatting ==="
	@buildifier --lint=warn --mode=check BUILD.bazel MODULE.bazel
	@echo "Bazel formatting OK"

# === Hardware Baseline Targets ===
# These targets measure and record the hardware capabilities of the current system
# to establish ground truth for benchmark comparisons.

HW_BASELINE_DIR := $(RESULTS_DIR)/hw-baseline

hw-info:
	@echo "=============================================="
	@echo "       HARDWARE CONFIGURATION SUMMARY"
	@echo "=============================================="
	@echo ""
	@echo "=== GPU Configuration ==="
	@nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total --format=csv
	@echo ""
	@echo "=== NVLink Topology ==="
	@nvidia-smi topo -m
	@echo ""
	@echo "=== NVLink Status ==="
	@nvidia-smi nvlink -s
	@echo ""
	@echo "=== CUDA/Driver Version ==="
	@nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | xargs -I{} echo "Driver: {}"
	@nvcc --version 2>/dev/null | grep "release" || echo "CUDA: (nvcc not in PATH)"
	@echo ""

# Full hardware baseline: captures all bandwidth measurements
hw-baseline: build-nvbandwidth
	@echo "=============================================="
	@echo "    HARDWARE BASELINE MEASUREMENT"
	@echo "=============================================="
	@echo ""
	@mkdir -p $(HW_BASELINE_DIR)
	@echo "Output directory: $(HW_BASELINE_DIR)"
	@echo ""
	@# Capture system info
	@echo "=== System Info ===" | tee $(HW_BASELINE_DIR)/system_info.txt
	@date | tee -a $(HW_BASELINE_DIR)/system_info.txt
	@hostname | tee -a $(HW_BASELINE_DIR)/system_info.txt
	@nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total --format=csv | tee -a $(HW_BASELINE_DIR)/system_info.txt
	@echo "" | tee -a $(HW_BASELINE_DIR)/system_info.txt
	@echo "=== NVLink Topology ===" | tee -a $(HW_BASELINE_DIR)/system_info.txt
	@nvidia-smi topo -m | tee -a $(HW_BASELINE_DIR)/system_info.txt
	@echo "" | tee -a $(HW_BASELINE_DIR)/system_info.txt
	@echo "=== NVLink Status ===" | tee -a $(HW_BASELINE_DIR)/system_info.txt
	@nvidia-smi nvlink -s | tee -a $(HW_BASELINE_DIR)/system_info.txt
	@echo ""
	@# Run nvbandwidth tests
	@echo "=== Running nvbandwidth D2D Tests ===" | tee $(HW_BASELINE_DIR)/nvbandwidth.txt
	@BAZEL_BIN=$$(bazel info bazel-bin) && \
	CUDA_VISIBLE_DEVICES=0,1 $$BAZEL_BIN/nvbandwidth \
		-t device_to_device_memcpy_read_ce \
		-t device_to_device_memcpy_write_ce \
		-t device_to_device_bidirectional_memcpy_read_ce \
		-t device_to_device_bidirectional_memcpy_write_ce \
		-t device_local_copy \
		2>&1 | tee -a $(HW_BASELINE_DIR)/nvbandwidth.txt
	@echo ""
	@# Run host bandwidth tests
	@echo "=== Running nvbandwidth Host Tests ===" | tee -a $(HW_BASELINE_DIR)/nvbandwidth.txt
	@BAZEL_BIN=$$(bazel info bazel-bin) && \
	CUDA_VISIBLE_DEVICES=0,1 $$BAZEL_BIN/nvbandwidth \
		-t host_to_device_memcpy_ce \
		-t device_to_host_memcpy_ce \
		2>&1 | tee -a $(HW_BASELINE_DIR)/nvbandwidth.txt
	@echo ""
	@# Run latency test
	@echo "=== Running nvbandwidth Latency Test ===" | tee -a $(HW_BASELINE_DIR)/nvbandwidth.txt
	@BAZEL_BIN=$$(bazel info bazel-bin) && \
	CUDA_VISIBLE_DEVICES=0,1 $$BAZEL_BIN/nvbandwidth \
		-t device_to_device_latency_sm \
		2>&1 | tee -a $(HW_BASELINE_DIR)/nvbandwidth.txt
	@echo ""
	@# Extract and summarize key metrics
	@echo "=============================================="
	@echo "           BASELINE SUMMARY"
	@echo "=============================================="
	@echo ""
	@echo "NVLink Configuration:"
	@nvidia-smi topo -m | grep -E "^GPU|NV[0-9]+" | head -3
	@echo ""
	@echo "Key Bandwidth Metrics (from nvbandwidth):"
	@grep -E "SUM device_to_device|SUM device_local" $(HW_BASELINE_DIR)/nvbandwidth.txt | \
		sed 's/SUM /  /' | head -10
	@echo ""
	@echo "D2D Latency:"
	@grep -A3 "Device to Device Latency" $(HW_BASELINE_DIR)/nvbandwidth.txt | tail -3
	@echo ""
	@echo "Host Stream:"
	@grep -E "SUM host_to_device|SUM device_to_host" $(HW_BASELINE_DIR)/nvbandwidth.txt | \
		sed 's/SUM /  /'
	@echo ""
	@echo "=============================================="
	@echo "Results saved to: $(HW_BASELINE_DIR)/"
	@echo "=============================================="

help:
	@echo "Yali Harness - Available targets:"
	@echo ""
	@echo "Setup:"
	@echo "  make setup        - Install deps, init submodules, create venv"
	@echo "  make deps         - Install system dependencies only"
	@echo "  make submodules   - Initialize git submodules only"
	@echo "  make venv         - Create Python virtual environment only"
	@echo ""
	@echo "Build:"
	@echo "  make build-all    - Build all binaries (yali, nccl, nvbandwidth, tests)"
	@echo "  make build-yali - Build Yali harness"
	@echo "  make build-yali-mpi - Build Yali MPI harness (requires MPI)"
	@echo "  make build-examples - Build single-process examples"
	@echo "  make build-examples-mpi - Build MPI examples (requires MPI)"
	@echo "  make build-nccl   - Build NCCL + nccl-tests"
	@echo "  make build-nvbandwidth - Build nvbandwidth"
	@echo "  make build-unit-tests - Build C++ unit tests"
	@echo ""
	@echo "Test (Single-Process):"
	@echo "  make test         - Run correctness tests"
	@echo "  make test-perf    - Run performance benchmarks"
	@echo "  make validate     - Run validation suite"
	@echo "  make test-unit-cpp - Build and run C++ unit tests (fast, standalone)"
	@echo "  make test-unit    - Run shell-based integration tests"
	@echo "  make test-examples - Run example correctness tests"
	@echo "  make test-ops     - Run ops API tests (correctness + perf)"
	@echo "  make test-all     - Run all tests (unit + correctness + examples + ops)"
	@echo ""
	@echo "Test (MPI Multi-Process):"
	@echo "  make test-mpi-correctness - Run MPI correctness tests"
	@echo "  make test-examples-mpi    - Run MPI example correctness tests"
	@echo "  make test-mpi-all         - Run all MPI tests"
	@echo ""
	@echo "Quick Run:"
	@echo "  make run-yali   - Run Yali @ 128M FP32"
	@echo "  make run-yali-mpi - Run Yali MPI mode (2 processes)"
	@echo "  make run-nccl     - Run NCCL all_reduce_perf @ 128M"
	@echo "  make run-nvbandwidth - Run nvbandwidth D2D test"
	@echo ""
	@echo "Utility:"
	@echo "  make clean        - Clean Bazel build and venv"
	@echo "  make clean-all    - Clean everything including NCCL build dirs"
	@echo "  make info         - Show build paths and GPU info"
	@echo "  make detect-arch  - Show detected GPU architecture"
	@echo ""
	@echo "Format/Lint:"
	@echo "  make setup-formatters - Install clang-format and buildifier"
	@echo "  make format       - Format all code (C++/CUDA + Bazel)"
	@echo "  make format-cpp   - Format C++/CUDA files only"
	@echo "  make format-bazel - Format Bazel files only"
	@echo "  make lint         - Check formatting without modifying"
	@echo "  make lint-cpp     - Check C++/CUDA formatting only"
	@echo "  make lint-bazel   - Check Bazel formatting only"
	@echo ""
	@echo "Hardware Baseline:"
	@echo "  make hw-info      - Show GPU/NVLink configuration summary"
	@echo "  make hw-baseline  - Run full nvbandwidth baseline (saves to RESULTS_DIR)"
	@echo ""
	@echo "GPU Architecture:"
	@echo "  Auto-detected: sm_$(CUDA_ARCH)"
	@echo "  Override:      make build-all CUDA_ARCH=90  (for H100)"
	@echo ""
	@echo "Quick start:"
	@echo "  make setup && source venv-2xa100/bin/activate && make build-all && make validate"
	@echo ""
	@echo "Benchmark Sweeps (all run BOTH single-process AND MPI):"
	@echo "  make sweep-quick            - Quick (~2 min): FP32 only, 5 sizes, cuda-events"
	@echo "  make sweep-standard         - Standard (~8 min): All dtypes, 11 sizes, cuda-events"
	@echo "  make sweep                  - Full (~10 min): Key sizes (16M/128M), all timing modes"
	@echo "  make sweep-extensive        - Extensive (~30 min): Key sizes, 10 runs, stability graphs"
	@echo "  make sweep-single-only      - Full sweep, skip MPI"
	@echo "  make sweep-mpi-only         - Full sweep, skip single-process"
	@echo "  make bench                  - Quick YALI vs NCCL comparison (5 sizes)"
	@echo ""
	@echo "Advanced Sweeps (NCCL modes):"
	@echo "  make sweep-nccl-1proc-1thr  - NCCL: -g 2 (single process, 2 GPUs)"
	@echo "  make sweep-nccl-1proc-2thr  - NCCL: -t 2 -g 1 (threaded)"
	@echo "  make sweep-nccl-2proc-mpi   - NCCL: mpirun -np 2 (MPI)"

# =============================================================================
# Main Sweep Targets (recommended)
# Uses scripts/sweep.py v2 for comprehensive benchmarking with statistics
# All sweeps now run BOTH single-process AND MPI modes in one run
# =============================================================================

# Quick sweep: FP32 only, 5 sizes, cuda-events (~2 min)
sweep-quick: build-all build-yali-mpi
	@. venv-2xa100/bin/activate && python3 scripts/sweep.py --quick

# Standard sweep: All dtypes, 11 sizes, cuda-events (~8 min)
sweep-standard: build-all build-yali-mpi
	@. venv-2xa100/bin/activate && python3 scripts/sweep.py --standard

# Full sweep: Key sizes (16M/128M), all timing modes (~10 min)
sweep: build-all build-yali-mpi
	@. venv-2xa100/bin/activate && python3 scripts/sweep.py

# Extensive sweep: Key sizes, 10 runs, stability graphs (~30 min)
sweep-extensive: build-all build-yali-mpi
	@. venv-2xa100/bin/activate && python3 scripts/sweep.py --extensive

# Single-process only (skip MPI)
sweep-single-only: build-all
	@. venv-2xa100/bin/activate && python3 scripts/sweep.py --single-only

# MPI only (skip single-process)
sweep-mpi-only: build-yali-mpi
	@. venv-2xa100/bin/activate && python3 scripts/sweep.py --mpi-only

# Quick benchmark comparison (uses quick_benchmark.py)
bench: build-yali
	@. venv-2xa100/bin/activate && python3 scripts/quick_benchmark.py

bench-mpi: build-yali-mpi
	@. venv-2xa100/bin/activate && python3 scripts/quick_benchmark.py --mpi

# =============================================================================
# NCCL Execution Modes:
#   Mode 1: Single-process, single-thread    (-g 2)         -> sweep-nccl-1proc-1thr
#   Mode 2: Single-process, thread-per-GPU   (-t 2 -g 1)    -> sweep-nccl-1proc-2thr
#   Mode 3: Multi-process (MPI)              (mpirun -np 2) -> sweep-nccl-2proc-mpi
# =============================================================================

# NCCL Mode 1: Single-process, single-thread, 2 GPUs per thread (-g 2)
sweep-nccl-1proc-1thr:
	@echo "=== NCCL Mode 1: Single-Process, Single-Thread (-g 2) ==="
	@mkdir -p $(RESULTS_DIR)/nccl-1proc-1thr
	@BAZEL_BIN=$$(bazel info bazel-bin) && \
	for algo in RING TREE; do \
		for dtype in float half bfloat16; do \
			echo "--- NCCL $$algo $$dtype ---"; \
			CUDA_VISIBLE_DEVICES=0,1 \
			LD_LIBRARY_PATH="$$BAZEL_BIN" \
			NCCL_ALGO=$$algo \
			$$BAZEL_BIN/all_reduce_perf -g 2 -b 2K -e 2G -f 2 -w 1 -n 5 -d $$dtype \
				2>&1 | tee $(RESULTS_DIR)/nccl-1proc-1thr/$$(echo $$algo | tr A-Z a-z)_$$dtype.txt; \
		done; \
	done
	@echo "--- NCCL DEVICE_API (Mode 1) ---"
	@BAZEL_BIN=$$(bazel info bazel-bin) && \
	for dtype in float half bfloat16; do \
		echo "--- NCCL DEVAPI $$dtype ---"; \
		CUDA_VISIBLE_DEVICES=0,1 \
		LD_LIBRARY_PATH="$$BAZEL_BIN" \
		$$BAZEL_BIN/all_reduce_perf -g 2 -b 2K -e 2G -f 2 -w 1 -n 5 -d $$dtype -D 2 -R 2 -z 1 \
			2>&1 | tee $(RESULTS_DIR)/nccl-1proc-1thr/devapi_$$dtype.txt; \
	done

# NCCL Mode 2: Single-process, thread-per-GPU (-t 2 -g 1)
sweep-nccl-1proc-2thr:
	@echo "=== NCCL Mode 2: Single-Process, Thread-per-GPU (-t 2 -g 1) ==="
	@mkdir -p $(RESULTS_DIR)/nccl-1proc-2thr
	@BAZEL_BIN=$$(bazel info bazel-bin) && \
	for algo in RING TREE; do \
		for dtype in float half bfloat16; do \
			echo "--- NCCL $$algo $$dtype (threaded) ---"; \
			CUDA_VISIBLE_DEVICES=0,1 \
			LD_LIBRARY_PATH="$$BAZEL_BIN" \
			NCCL_ALGO=$$algo \
			$$BAZEL_BIN/all_reduce_perf -t 2 -g 1 -b 2K -e 2G -f 2 -w 1 -n 5 -d $$dtype \
				2>&1 | tee $(RESULTS_DIR)/nccl-1proc-2thr/$$(echo $$algo | tr A-Z a-z)_$$dtype.txt; \
		done; \
	done
	@echo "--- NCCL DEVICE_API (Mode 2) ---"
	@BAZEL_BIN=$$(bazel info bazel-bin) && \
	for dtype in float half bfloat16; do \
		echo "--- NCCL DEVAPI $$dtype (threaded) ---"; \
		CUDA_VISIBLE_DEVICES=0,1 \
		LD_LIBRARY_PATH="$$BAZEL_BIN" \
		$$BAZEL_BIN/all_reduce_perf -t 2 -g 1 -b 2K -e 2G -f 2 -w 1 -n 5 -d $$dtype -D 2 -R 2 -z 1 \
			2>&1 | tee $(RESULTS_DIR)/nccl-1proc-2thr/devapi_$$dtype.txt; \
	done

# NCCL Mode 3: Multi-process with MPI (mpirun -np 2, 1 GPU per process)
# Requires: apt-get install openmpi-bin libopenmpi-dev
# Build:    make build-nccl-mpi
# Note: NCCL/nccl-tests auto-assigns GPUs based on local rank when -g 1 is used
NCCL_MPI_LIB := $(CURDIR)/nccl/build/lib
NCCL_MPI_BIN := $(CURDIR)/nccl-tests/build/all_reduce_perf

sweep-nccl-2proc-mpi:
	@echo "=== NCCL Mode 3: Multi-Process MPI (mpirun -np 2 -g 1) ==="
	@if ! command -v mpirun >/dev/null 2>&1; then \
		echo "ERROR: mpirun not found. Run: make setup-mpi"; \
		exit 1; \
	fi
	@if [ ! -f "$(NCCL_MPI_BIN)" ]; then \
		echo "Building MPI-enabled nccl-tests..."; \
		$(MAKE) build-nccl-mpi; \
	fi
	@mkdir -p $(RESULTS_DIR)/nccl-2proc-mpi
	@for algo in RING TREE; do \
		for dtype in float half bfloat16; do \
			echo "--- NCCL $$algo $$dtype (MPI) ---"; \
			CUDA_VISIBLE_DEVICES=0,1 \
			LD_LIBRARY_PATH="$(NCCL_MPI_LIB)" \
			NCCL_ALGO=$$algo \
			mpirun -np 2 --allow-run-as-root \
				-x LD_LIBRARY_PATH \
				-x NCCL_ALGO \
				-x CUDA_VISIBLE_DEVICES \
				--bind-to none \
				$(NCCL_MPI_BIN) -g 1 -b 2K -e 2G -f 2 -w 1 -n 5 -d $$dtype \
				2>&1 | tee $(RESULTS_DIR)/nccl-2proc-mpi/$$(echo $$algo | tr A-Z a-z)_$$dtype.txt; \
		done; \
	done
	@echo "--- NCCL DEVICE_API (Mode 3) ---"
	@for dtype in float half bfloat16; do \
		echo "--- NCCL DEVAPI $$dtype (MPI) ---"; \
		CUDA_VISIBLE_DEVICES=0,1 \
		LD_LIBRARY_PATH="$(NCCL_MPI_LIB)" \
		mpirun -np 2 --allow-run-as-root \
			-x LD_LIBRARY_PATH \
			-x CUDA_VISIBLE_DEVICES \
			--bind-to none \
			$(NCCL_MPI_BIN) -g 1 -b 2K -e 2G -f 2 -w 1 -n 5 -d $$dtype -D 2 -R 2 -z 1 \
			2>&1 | tee $(RESULTS_DIR)/nccl-2proc-mpi/devapi_$$dtype.txt; \
	done

# Run all NCCL execution modes
sweep-nccl-all-modes: sweep-nccl-1proc-1thr sweep-nccl-1proc-2thr
	@echo ""
	@echo "=== NCCL All Modes Complete ==="
	@echo "Mode 1 (1proc-1thr): $(RESULTS_DIR)/nccl-1proc-1thr/"
	@echo "Mode 2 (1proc-2thr): $(RESULTS_DIR)/nccl-1proc-2thr/"
	@if command -v mpirun >/dev/null 2>&1; then \
		$(MAKE) sweep-nccl-2proc-mpi RESULTS_DIR=$(RESULTS_DIR); \
		echo "Mode 3 (2proc-mpi):  $(RESULTS_DIR)/nccl-2proc-mpi/"; \
	else \
		echo "Mode 3 (2proc-mpi):  SKIPPED (MPI not available)"; \
	fi


