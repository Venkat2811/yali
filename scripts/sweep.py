#!/usr/bin/env python3
"""
YALI vs NCCL Comprehensive Benchmark Sweep v2

Single-command benchmark suite with proper statistical methodology.
Runs system discovery, hardware baseline, examples, and full benchmark sweeps.
Generates graphs and detailed summary reports.

Features:
- Three modes: Quick (~5 min), Full (~20 min), Extensive (~90 min)
- Both single-process AND MPI modes in one sweep
- All timing modes: cuda-events, throughput, latency, profiler
- All dtypes: FP32, FP16, BF16
- tqdm progress bars for overall and per-task progress
- Comprehensive graph generation (bandwidth, speedup, stability, latency)
- Detailed summary.md with embedded graphs and tables

Usage:
    python scripts/sweep.py --quick      # Fast sanity check (~5 min)
    python scripts/sweep.py              # Full benchmark (~20 min)
    python scripts/sweep.py --extensive  # Statistical rigor (~90 min)
"""

import argparse
import csv
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from tabulate import tabulate
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Constants
# =============================================================================

SIZE_BYTES = {
    "4K": 4 * 1024,
    "16K": 16 * 1024,
    "64K": 64 * 1024,
    "256K": 256 * 1024,
    "1M": 1024 * 1024,
    "4M": 4 * 1024 * 1024,
    "16M": 16 * 1024 * 1024,
    "64M": 64 * 1024 * 1024,
    "128M": 128 * 1024 * 1024,
    "256M": 256 * 1024 * 1024,
    "512M": 512 * 1024 * 1024,
    "1G": 1024 * 1024 * 1024,
    "2G": 2 * 1024 * 1024 * 1024,
}

# Sweep configurations
# Size sets
ALL_SIZES = ["4K", "64K", "1M", "4M", "16M", "64M", "128M", "256M", "512M", "1G", "2G"]  # 11 sizes
KEY_SIZES = ["16M", "128M"]  # Most representative sizes for detailed analysis
QUICK_SIZES = ["4K", "16M", "64M", "128M", "2G"]  # 5 sizes for sanity check

# Timing modes
CUDA_EVENTS_ONLY = ["cuda-events"]
ALL_TIMING = ["cuda-events", "throughput", "latency"]

# Dtype sets
ALL_DTYPES = {"fp32": 4, "fp16": 2, "bf16": 2}
FP32_ONLY = {"fp32": 4}

# Execution modes
EXEC_MODES = ["single", "mpi"]

# Colors for graphs
COLORS = {
    "yali_single": "#2ecc71",   # Green
    "yali_mpi": "#27ae60",      # Dark green
    "nccl_single": "#3498db",   # Blue
    "nccl_mpi": "#2980b9",      # Dark blue
    "speedup_pos": "#27ae60",   # Dark green
    "speedup_neg": "#e74c3c",   # Red
}


# =============================================================================
# Helper Functions
# =============================================================================

def bytes_to_human(b: int) -> str:
    if b >= 1024**3:
        return f"{b // (1024**3)} GB"
    elif b >= 1024**2:
        return f"{b // (1024**2)} MB"
    elif b >= 1024:
        return f"{b // 1024} KB"
    return f"{b} B"


def elems_to_human(e: int) -> str:
    if e >= 1024**3:
        return f"{e // (1024**3)}G"
    elif e >= 1024**2:
        return f"{e // (1024**2)}M"
    elif e >= 1024:
        return f"{e // 1024}K"
    return str(e)


def format_speedup(speedup: float) -> str:
    """Format speedup as both x factor and percentage."""
    pct = (speedup - 1) * 100
    sign = "+" if pct >= 0 else ""
    return f"{speedup:.2f}x ({sign}{pct:.0f}%)"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchStats:
    """Statistics from multiple benchmark runs."""
    mean: float
    stddev: float
    min_val: float
    max_val: float
    samples: List[float]
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0

    @classmethod
    def from_samples(cls, samples: List[float]) -> Optional['BenchStats']:
        if not samples:
            return None
        n = len(samples)
        mean = sum(samples) / n
        if n > 1:
            variance = sum((x - mean) ** 2 for x in samples) / (n - 1)
            stddev = math.sqrt(variance)
        else:
            stddev = 0.0

        sorted_samples = sorted(samples)
        p50 = sorted_samples[int(n * 0.50)] if n > 0 else mean
        p90 = sorted_samples[min(int(n * 0.90), n-1)] if n > 0 else mean
        p95 = sorted_samples[min(int(n * 0.95), n-1)] if n > 0 else mean
        p99 = sorted_samples[min(int(n * 0.99), n-1)] if n > 0 else mean

        return cls(
            mean=mean,
            stddev=stddev,
            min_val=min(samples),
            max_val=max(samples),
            samples=samples,
            p50=p50,
            p90=p90,
            p95=p95,
            p99=p99
        )

    def cv_percent(self) -> float:
        """Coefficient of variation as percentage."""
        return (self.stddev / self.mean * 100) if self.mean > 0 else 0

    def format(self) -> str:
        if self.stddev < 0.01:
            return f"{self.mean:.2f}"
        return f"{self.mean:.1f}±{self.stddev:.1f}"


@dataclass
class SweepConfig:
    """Configuration for a sweep run."""
    mode: str  # quick, full, extensive
    sizes: List[str]
    timing_modes: List[str]
    dtypes: Dict[str, int]
    runs: int
    calls: int
    exec_modes: List[str]  # single, mpi, or both
    with_profiler: bool = False

    @classmethod
    def quick(cls) -> 'SweepConfig':
        """Quick sanity check: FP32 only, 5 sizes, cuda-events (~2 min)"""
        return cls(
            mode="quick",
            sizes=QUICK_SIZES,
            timing_modes=CUDA_EVENTS_ONLY,
            dtypes=FP32_ONLY,
            runs=2,
            calls=20,
            exec_modes=EXEC_MODES,
            with_profiler=False
        )

    @classmethod
    def standard(cls) -> 'SweepConfig':
        """Standard sweep: All sizes, all dtypes, cuda-events only (~8 min)"""
        return cls(
            mode="standard",
            sizes=ALL_SIZES,
            timing_modes=CUDA_EVENTS_ONLY,
            dtypes=ALL_DTYPES,
            runs=2,
            calls=20,
            exec_modes=EXEC_MODES,
            with_profiler=False
        )

    @classmethod
    def full(cls) -> 'SweepConfig':
        """Full sweep: Key sizes (16M, 128M), all timing modes (~10 min)"""
        return cls(
            mode="full",
            sizes=KEY_SIZES,
            timing_modes=ALL_TIMING,
            dtypes=ALL_DTYPES,
            runs=3,
            calls=20,
            exec_modes=EXEC_MODES,
            with_profiler=False
        )

    @classmethod
    def extensive(cls) -> 'SweepConfig':
        """Extensive sweep: Key sizes, all timing modes, 10 runs (~30 min)"""
        return cls(
            mode="extensive",
            sizes=KEY_SIZES,
            timing_modes=ALL_TIMING,
            dtypes=ALL_DTYPES,
            runs=10,
            calls=20,
            exec_modes=EXEC_MODES,
            with_profiler=True
        )

    def total_benchmarks(self) -> int:
        """Calculate total number of benchmark invocations."""
        return (len(self.exec_modes) * 2 * len(self.dtypes) *
                len(self.timing_modes) * len(self.sizes) * self.runs)


# =============================================================================
# Progress Tracker
# =============================================================================

class ProgressTracker:
    """Manages tqdm progress bars for the sweep."""

    def __init__(self, config: SweepConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.start_time = datetime.now()
        self.phase_times: Dict[str, float] = {}
        self.current_phase = ""

        # Progress bars
        self.overall_pbar: Optional[tqdm] = None
        self.task_pbar: Optional[tqdm] = None

        # Phases for overall progress
        self.phases = [
            "Build",
            "System Info",
            "HW Baseline",
            "Examples",
        ]
        # Add benchmark phases for each exec mode
        for exec_mode in config.exec_modes:
            self.phases.append(f"YALI {exec_mode.title()}")
            self.phases.append(f"NCCL {exec_mode.title()}")
        self.phases.extend(["Graphs", "Summary"])

        self.phase_idx = 0

    def start(self):
        """Initialize progress bars."""
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"  YALI vs NCCL Benchmark Sweep")
            print(f"  Mode: {self.config.mode.title()} | "
                  f"Dtypes: {', '.join(d.upper() for d in self.config.dtypes)} | "
                  f"Sizes: {len(self.config.sizes)}")
            print("=" * 70 + "\n")

            self.overall_pbar = tqdm(
                total=len(self.phases),
                desc="Sweep Progress",
                position=0,
                leave=True,
                bar_format='{desc}: {bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )

    def start_phase(self, phase: str):
        """Start a new phase."""
        self.current_phase = phase
        self.phase_times[phase] = datetime.now().timestamp()
        if self.overall_pbar:
            self.overall_pbar.set_description(f"Sweep [{phase}]")

    def end_phase(self):
        """End current phase and update progress."""
        if self.current_phase and self.current_phase in self.phase_times:
            elapsed = datetime.now().timestamp() - self.phase_times[self.current_phase]
            self.phase_times[self.current_phase] = elapsed
        if self.overall_pbar:
            self.overall_pbar.update(1)
        self.phase_idx += 1

    def start_benchmark_task(self, total: int, desc: str):
        """Start a benchmark task with its own progress bar."""
        if self.task_pbar:
            self.task_pbar.close()
        if self.verbose:
            self.task_pbar = tqdm(
                total=total,
                desc=desc,
                position=1,
                leave=False,
                bar_format='{desc}: {bar}| {n_fmt}/{total_fmt} [{rate_fmt}]'
            )

    def update_benchmark(self, result_str: str = ""):
        """Update benchmark progress."""
        if self.task_pbar:
            self.task_pbar.update(1)
            if result_str:
                self.task_pbar.set_postfix_str(result_str)

    def end_benchmark_task(self):
        """Close benchmark task progress bar."""
        if self.task_pbar:
            self.task_pbar.close()
            self.task_pbar = None

    def finish(self):
        """Finish all progress tracking."""
        if self.task_pbar:
            self.task_pbar.close()
        if self.overall_pbar:
            self.overall_pbar.close()

        total_time = datetime.now() - self.start_time

        if self.verbose:
            print("\n" + "-" * 50)
            print(f"{'Phase':<20} {'Time':>10}")
            print("-" * 50)
            for phase in self.phases:
                if phase in self.phase_times:
                    t = self.phase_times[phase]
                    print(f"{phase:<20} {t:>8.1f}s")
            print("-" * 50)
            print(f"{'Total':<20} {total_time.total_seconds():>8.1f}s")
            print("-" * 50)


# =============================================================================
# Sweep Runner
# =============================================================================

class SweepRunner:
    """Orchestrates the comprehensive benchmark sweep."""

    def __init__(self, output_dir: Path, config: SweepConfig, verbose: bool = True):
        self.output_dir = output_dir
        self.config = config
        self.verbose = verbose
        self.bazel_bin = None
        self.d2d_bw = 93.70  # Default, updated by nvbandwidth

        # Results storage: results[exec_mode][library][dtype][timing][size] = BenchStats
        self.results: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, BenchStats]]]]] = {}

        # Progress tracker
        self.progress = ProgressTracker(config, verbose)

        # Create output directories
        self._create_directories()

    def _create_directories(self):
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "hw-baseline").mkdir(exist_ok=True)
        (self.output_dir / "examples").mkdir(exist_ok=True)
        (self.output_dir / "graphs").mkdir(exist_ok=True)

        for dtype in self.config.dtypes:
            (self.output_dir / "graphs" / dtype).mkdir(exist_ok=True)

        for exec_mode in self.config.exec_modes:
            mode_dir = "single-process" if exec_mode == "single" else "mpi"
            (self.output_dir / mode_dir / "yali").mkdir(parents=True, exist_ok=True)
            (self.output_dir / mode_dir / "nccl").mkdir(parents=True, exist_ok=True)

        if self.config.with_profiler:
            (self.output_dir / "profiler").mkdir(exist_ok=True)

    def run_cmd(self, cmd: List[str], env: dict = None, timeout: int = 300) -> subprocess.CompletedProcess:
        """Run command and return result."""
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        try:
            return subprocess.run(cmd, capture_output=True, text=True, env=full_env, timeout=timeout)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(cmd, 1, "", "timeout")
        except FileNotFoundError as e:
            return subprocess.CompletedProcess(cmd, 1, "", str(e))

    def get_bazel_bin(self) -> str:
        if self.bazel_bin:
            return self.bazel_bin
        result = self.run_cmd(["bazel", "info", "bazel-bin"])
        self.bazel_bin = result.stdout.strip()
        return self.bazel_bin

    def sol_pct(self, bw: float) -> float:
        """Calculate Speed-of-Light percentage."""
        return (bw / self.d2d_bw) * 100 if self.d2d_bw > 0 else 0

    # =========================================================================
    # Phase 1: Build
    # =========================================================================
    def build_targets(self) -> bool:
        self.progress.start_phase("Build")

        targets = ["//:benchmark_yali", "//:benchmark_nccl", "//:nvbandwidth_bin",
                   "//:example_simple", "//:example_multilane", "//:test_ops_allreduce"]

        if "mpi" in self.config.exec_modes:
            targets.extend(["//:benchmark_yali_mpi", "//:benchmark_nccl_mpi",
                           "//:example_simple_mpi", "//:example_multilane_mpi"])

        cmd = ["bazel", "build"] + targets
        result = self.run_cmd(cmd, timeout=600)

        self.progress.end_phase()
        return result.returncode == 0

    # =========================================================================
    # Phase 2: System Info
    # =========================================================================
    def collect_system_info(self) -> Dict[str, Any]:
        self.progress.start_phase("System Info")

        info = {
            "timestamp": datetime.now().isoformat(),
            "hostname": os.uname().nodename,
            "gpus": [],
            "nvlink": {},
            "cuda_version": "",
            "driver_version": "",
            "sweep_config": {
                "mode": self.config.mode,
                "sizes": self.config.sizes,
                "timing_modes": self.config.timing_modes,
                "dtypes": list(self.config.dtypes.keys()),
                "runs": self.config.runs,
                "calls": self.config.calls,
                "exec_modes": self.config.exec_modes,
            }
        }

        # GPU info
        result = self.run_cmd(["nvidia-smi", "--query-gpu=index,name,pci.bus_id,memory.total,compute_cap",
                               "--format=csv,noheader"])
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    info["gpus"].append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "pci_bus": parts[2],
                        "memory": parts[3],
                        "compute_cap": parts[4],
                    })

        # NVLink topology
        result = self.run_cmd(["nvidia-smi", "topo", "-m"])
        if result.returncode == 0:
            info["nvlink"]["topology"] = result.stdout
            match = re.search(r'NV(\d+)', result.stdout)
            if match:
                info["nvlink"]["version"] = f"NV{match.group(1)}"
                info["nvlink"]["links"] = int(match.group(1))

        # Save
        with open(self.output_dir / "hw-baseline" / "system_info.json", "w") as f:
            json.dump(info, f, indent=2)

        self.progress.end_phase()
        return info

    # =========================================================================
    # Phase 3: Hardware Baseline
    # =========================================================================
    def run_nvbandwidth(self) -> Dict[str, float]:
        self.progress.start_phase("HW Baseline")

        bazel_bin = self.get_bazel_bin()
        nvbw_bin = f"{bazel_bin}/nvbandwidth"
        results = {"d2d_unidirectional": 93.70, "d2d_bidirectional": 187.40}

        if Path(nvbw_bin).exists():
            env = {"CUDA_VISIBLE_DEVICES": "0,1"}

            # Unidirectional
            result = self.run_cmd([nvbw_bin, "-t", "device_to_device_memcpy_read_ce"], env=env)
            with open(self.output_dir / "hw-baseline" / "nvbandwidth_d2d.txt", "w") as f:
                f.write(result.stdout)
            match = re.search(r'^\s*0\s+N/A\s+([0-9.]+)', result.stdout, re.MULTILINE)
            if match:
                results["d2d_unidirectional"] = float(match.group(1))

            # Bidirectional
            result = self.run_cmd([nvbw_bin, "-t", "device_to_device_bidirectional_memcpy_read_ce"], env=env)
            with open(self.output_dir / "hw-baseline" / "nvbandwidth_bidir.txt", "w") as f:
                f.write(result.stdout)
            match = re.search(r'SUM\s+\S+\s+([0-9.]+)', result.stdout)
            if match:
                results["d2d_bidirectional"] = float(match.group(1))

        self.d2d_bw = results["d2d_unidirectional"]
        self.progress.end_phase()
        return results

    # =========================================================================
    # Phase 4: Examples
    # =========================================================================
    def run_examples(self) -> Dict[str, bool]:
        self.progress.start_phase("Examples")

        bazel_bin = self.get_bazel_bin()
        env = {"CUDA_VISIBLE_DEVICES": "0,1"}
        results = {}

        examples = [("simple", f"{bazel_bin}/example_simple"),
                    ("multilane", f"{bazel_bin}/example_multilane")]

        if "mpi" in self.config.exec_modes:
            examples.extend([
                ("simple_mpi", f"mpirun -np 2 --allow-run-as-root --bind-to none -x CUDA_VISIBLE_DEVICES {bazel_bin}/example_simple_mpi"),
                ("multilane_mpi", f"mpirun -np 2 --allow-run-as-root --bind-to none -x CUDA_VISIBLE_DEVICES {bazel_bin}/example_multilane_mpi"),
            ])

        for name, cmd in examples:
            result = self.run_cmd(cmd.split(), env=env)
            results[name] = result.returncode == 0

        # Save results
        with open(self.output_dir / "examples" / "results.txt", "w") as f:
            for name, passed in results.items():
                f.write(f"{name}: {'PASS' if passed else 'FAIL'}\n")

        self.progress.end_phase()
        return results

    # =========================================================================
    # Phase 5: Benchmarks
    # =========================================================================
    def run_single_benchmark(self, exec_mode: str, bench_type: str, elements: int,
                              timing_mode: str, dtype: str) -> Optional[float]:
        """Run a single benchmark and return GB/s."""
        bazel_bin = self.get_bazel_bin()
        binary = f"benchmark_{bench_type.lower()}"
        if exec_mode == "mpi":
            binary += "_mpi"

        # Single-process:
        #   YALI args: elements, calls, verify, mode, lanes, timing, dtype
        #   NCCL args: elements, calls, timing, dtype
        # MPI:
        #   YALI MPI args: elements, calls, verify, mode, lanes, timing, dtype
        #   NCCL MPI args: elements, calls, timing, dtype
        if exec_mode == "mpi":
            if bench_type == "yali":
                args = [str(elements), str(self.config.calls), "0", "auto", "0", timing_mode, dtype]
            else:  # nccl
                args = [str(elements), str(self.config.calls), timing_mode, dtype]
        else:
            if bench_type == "yali":
                args = [str(elements), str(self.config.calls), "0", "auto", "0", timing_mode, dtype]
            else:  # nccl
                args = [str(elements), str(self.config.calls), timing_mode, dtype]

        if exec_mode == "mpi":
            cmd = ["mpirun", "-np", "2", "--allow-run-as-root", "--bind-to", "none",
                   "-x", "CUDA_VISIBLE_DEVICES"]
            if bench_type == "nccl":
                cmd.extend(["-x", "LD_LIBRARY_PATH"])
            cmd.extend([f"{bazel_bin}/{binary}"] + args)
        else:
            cmd = [f"{bazel_bin}/{binary}"] + args

        env = {"CUDA_VISIBLE_DEVICES": "0,1"}
        if bench_type == "nccl":
            env["LD_LIBRARY_PATH"] = f"third_party/nccl/build/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

        result = self.run_cmd(cmd, env=env, timeout=180)

        for line in result.stdout.split("\n"):
            if "GB/s" in line and bench_type.upper() in line:
                match = re.search(r'([\d.]+)\s*GB/s', line)
                if match:
                    return float(match.group(1))
        return None

    def run_benchmark_sweep(self, exec_mode: str, bench_type: str) -> Dict[str, Dict[str, Dict[str, BenchStats]]]:
        """Run benchmark sweep for one exec_mode and bench_type."""
        phase_name = f"{bench_type.upper()} {exec_mode.title()}"
        self.progress.start_phase(phase_name)

        total_tasks = len(self.config.dtypes) * len(self.config.timing_modes) * len(self.config.sizes) * self.config.runs
        self.progress.start_benchmark_task(total_tasks, f"{bench_type.upper()} {exec_mode}")

        all_results: Dict[str, Dict[str, Dict[str, BenchStats]]] = {}

        for dtype, elem_size in self.config.dtypes.items():
            all_results[dtype] = {}
            csv_data = []

            for timing_mode in self.config.timing_modes:
                all_results[dtype][timing_mode] = {}

                for size_name in self.config.sizes:
                    size_bytes = SIZE_BYTES[size_name]
                    elements = size_bytes // elem_size

                    samples = []
                    for run in range(self.config.runs):
                        gbps = self.run_single_benchmark(exec_mode, bench_type, elements, timing_mode, dtype)
                        if gbps is not None:
                            samples.append(gbps)
                        self.progress.update_benchmark(f"{size_name}: {gbps:.1f} GB/s" if gbps else "")

                    stats = BenchStats.from_samples(samples)
                    if stats:
                        all_results[dtype][timing_mode][size_name] = stats
                        csv_data.append({
                            "timing_mode": timing_mode,
                            "size": size_name,
                            "bytes": size_bytes,
                            "elements": elements,
                            "mean_gbps": stats.mean,
                            "stddev_gbps": stats.stddev,
                            "min_gbps": stats.min_val,
                            "max_gbps": stats.max_val,
                            "cv_pct": stats.cv_percent(),
                            "p50": stats.p50,
                            "p90": stats.p90,
                            "p95": stats.p95,
                            "p99": stats.p99,
                            "sol_pct": self.sol_pct(stats.mean),
                            "runs": len(stats.samples),
                            "samples": ",".join(f"{s:.2f}" for s in stats.samples),
                        })

            # Save CSV
            mode_dir = "single-process" if exec_mode == "single" else "mpi"
            csv_file = self.output_dir / mode_dir / bench_type.lower() / f"{dtype}.csv"
            if csv_data:
                with open(csv_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                    writer.writeheader()
                    writer.writerows(csv_data)

        self.progress.end_benchmark_task()
        self.progress.end_phase()
        return all_results

    def run_all_benchmarks(self):
        """Run all benchmark sweeps."""
        for exec_mode in self.config.exec_modes:
            # Initialize storage
            if exec_mode not in self.results:
                self.results[exec_mode] = {}

            # YALI
            self.results[exec_mode]["yali"] = self.run_benchmark_sweep(exec_mode, "yali")

            # NCCL
            self.results[exec_mode]["nccl"] = self.run_benchmark_sweep(exec_mode, "nccl")

    # =========================================================================
    # Phase 6: Graph Generation
    # =========================================================================
    def generate_graphs(self):
        """Generate all comparison graphs."""
        self.progress.start_phase("Graphs")

        for dtype in self.config.dtypes:
            self._generate_bandwidth_graph(dtype)
            self._generate_speedup_graph(dtype)
            self._generate_speedup_percentage_graph(dtype)

            if self.config.mode in ["full", "extensive"]:
                self._generate_timing_comparison_graph(dtype)

            if self.config.mode == "extensive":
                self._generate_variance_errorbar_graph(dtype)
                self._generate_variance_coefficient_graph(dtype)
                self._generate_min_max_spread_graph(dtype)
                self._generate_latency_boxplot(dtype)
                self._generate_latency_percentiles_graph(dtype)
                self._generate_latency_cdf_graph(dtype)

        self._generate_executive_summary_graph()

        if self.config.mode in ["full", "extensive"]:
            self._generate_speedup_heatmap()

        self.progress.end_phase()

    def _get_data_for_graph(self, dtype: str, timing: str = "cuda-events") -> Tuple[List[str], Dict]:
        """Extract data for graphing."""
        sizes = [s for s in self.config.sizes]
        data = {}

        for exec_mode in self.config.exec_modes:
            for lib in ["yali", "nccl"]:
                key = f"{lib}_{exec_mode}"
                data[key] = []
                for size in sizes:
                    stats = self.results.get(exec_mode, {}).get(lib, {}).get(dtype, {}).get(timing, {}).get(size)
                    data[key].append(stats.mean if stats else 0)

        return sizes, data

    def _generate_bandwidth_graph(self, dtype: str):
        """Generate bandwidth comparison graph."""
        sizes, data = self._get_data_for_graph(dtype)

        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(sizes))
        width = 0.2

        bars = []
        labels = []
        for i, (key, color) in enumerate([
            ("yali_single", COLORS["yali_single"]),
            ("nccl_single", COLORS["nccl_single"]),
            ("yali_mpi", COLORS["yali_mpi"]) if "mpi" in self.config.exec_modes else (None, None),
            ("nccl_mpi", COLORS["nccl_mpi"]) if "mpi" in self.config.exec_modes else (None, None),
        ]):
            if key and key in data:
                offset = (i - 1.5) * width if "mpi" in self.config.exec_modes else (i - 0.5) * width
                bar = ax.bar(x + offset, data[key], width, label=key.replace("_", " ").title(),
                           color=color, edgecolor='white')
                bars.append(bar)
                labels.append(key)

        # SoL reference line
        ax.axhline(y=self.d2d_bw, color='gray', linestyle='--', alpha=0.7,
                   label=f'SoL ({self.d2d_bw:.0f} GB/s)')

        ax.set_xlabel('Message Size', fontsize=12)
        ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
        ax.set_title(f'AllReduce Bandwidth: YALI vs NCCL ({dtype.upper()})', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, rotation=45, ha='right')
        ax.legend(loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "graphs" / dtype / "bandwidth_comparison.png", dpi=150)
        plt.close()

    def _generate_speedup_graph(self, dtype: str):
        """Generate speedup comparison graph with x factor and percentage."""
        sizes, data = self._get_data_for_graph(dtype)

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(sizes))
        width = 0.35 if "mpi" in self.config.exec_modes else 0.6

        # Calculate speedups
        single_speedup = [y/n if n > 0 else 1 for y, n in zip(data.get("yali_single", [0]*len(sizes)),
                                                               data.get("nccl_single", [1]*len(sizes)))]

        colors_single = [COLORS["speedup_pos"] if s >= 1 else COLORS["speedup_neg"] for s in single_speedup]
        bars1 = ax.bar(x - width/2 if "mpi" in self.config.exec_modes else x, single_speedup,
                       width, label='Single-Process', color=colors_single, edgecolor='white', alpha=0.8)

        if "mpi" in self.config.exec_modes:
            mpi_speedup = [y/n if n > 0 else 1 for y, n in zip(data.get("yali_mpi", [0]*len(sizes)),
                                                                data.get("nccl_mpi", [1]*len(sizes)))]
            colors_mpi = [COLORS["speedup_pos"] if s >= 1 else COLORS["speedup_neg"] for s in mpi_speedup]
            bars2 = ax.bar(x + width/2, mpi_speedup, width, label='MPI',
                          color=colors_mpi, edgecolor='white', alpha=0.6, hatch='//')

        # Reference line at 1.0
        ax.axhline(y=1.0, color='black', linestyle='-', linewidth=2)

        # Add labels with x factor and %
        for i, (bar, spd) in enumerate(zip(bars1, single_speedup)):
            pct = (spd - 1) * 100
            sign = "+" if pct >= 0 else ""
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{spd:.2f}x\n({sign}{pct:.0f}%)', ha='center', va='bottom', fontsize=7, fontweight='bold')

        ax.set_xlabel('Message Size', fontsize=12)
        ax.set_ylabel('Speedup (YALI / NCCL)', fontsize=12)
        ax.set_title(f'YALI Speedup over NCCL ({dtype.upper()})', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(single_speedup) * 1.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "graphs" / dtype / "speedup_by_mode.png", dpi=150)
        plt.close()

    def _generate_speedup_percentage_graph(self, dtype: str):
        """Generate percentage improvement graph."""
        sizes, data = self._get_data_for_graph(dtype)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(sizes))
        width = 0.35 if "mpi" in self.config.exec_modes else 0.6

        # Calculate % improvement
        single_pct = [(y/n - 1) * 100 if n > 0 else 0 for y, n in
                      zip(data.get("yali_single", [0]*len(sizes)), data.get("nccl_single", [1]*len(sizes)))]

        colors = [COLORS["speedup_pos"] if p >= 0 else COLORS["speedup_neg"] for p in single_pct]
        bars1 = ax.bar(x - width/2 if "mpi" in self.config.exec_modes else x, single_pct,
                       width, label='Single-Process', color=colors, edgecolor='white')

        if "mpi" in self.config.exec_modes:
            mpi_pct = [(y/n - 1) * 100 if n > 0 else 0 for y, n in
                       zip(data.get("yali_mpi", [0]*len(sizes)), data.get("nccl_mpi", [1]*len(sizes)))]
            colors_mpi = [COLORS["speedup_pos"] if p >= 0 else COLORS["speedup_neg"] for p in mpi_pct]
            ax.bar(x + width/2, mpi_pct, width, label='MPI', color=colors_mpi, edgecolor='white', alpha=0.7)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

        # Add labels
        for bar, pct in zip(bars1, single_pct):
            sign = "+" if pct >= 0 else ""
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (2 if pct >= 0 else -5),
                    f'{sign}{pct:.0f}%', ha='center', va='bottom' if pct >= 0 else 'top', fontsize=8)

        ax.set_xlabel('Message Size', fontsize=12)
        ax.set_ylabel('Improvement (%)', fontsize=12)
        ax.set_title(f'YALI Improvement over NCCL ({dtype.upper()})', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "graphs" / dtype / "speedup_percentage.png", dpi=150)
        plt.close()

    def _generate_timing_comparison_graph(self, dtype: str):
        """Generate timing mode comparison graph."""
        sizes = self.config.sizes

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, exec_mode in zip(axes, self.config.exec_modes[:2]):
            for timing, linestyle in [("cuda-events", "-"), ("throughput", "--"), ("latency", ":")]:
                if timing not in self.config.timing_modes:
                    continue
                yali_data = []
                nccl_data = []
                for size in sizes:
                    yali_stats = self.results.get(exec_mode, {}).get("yali", {}).get(dtype, {}).get(timing, {}).get(size)
                    nccl_stats = self.results.get(exec_mode, {}).get("nccl", {}).get(dtype, {}).get(timing, {}).get(size)
                    yali_data.append(yali_stats.mean if yali_stats else 0)
                    nccl_data.append(nccl_stats.mean if nccl_stats else 0)

                ax.plot(range(len(sizes)), yali_data, linestyle=linestyle, marker='o',
                       label=f'YALI {timing}', color=COLORS["yali_single"])
                ax.plot(range(len(sizes)), nccl_data, linestyle=linestyle, marker='s',
                       label=f'NCCL {timing}', color=COLORS["nccl_single"])

            ax.set_xlabel('Message Size')
            ax.set_ylabel('Bandwidth (GB/s)')
            ax.set_title(f'{exec_mode.title()} Mode')
            ax.set_xticks(range(len(sizes)))
            ax.set_xticklabels(sizes, rotation=45, ha='right')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.suptitle(f'Timing Mode Comparison ({dtype.upper()})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "graphs" / dtype / "timing_mode_comparison.png", dpi=150)
        plt.close()

    def _generate_variance_errorbar_graph(self, dtype: str):
        """Generate bandwidth with error bars graph."""
        sizes, _ = self._get_data_for_graph(dtype)

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(sizes))
        width = 0.35

        for exec_mode in self.config.exec_modes[:1]:  # Just single for clarity
            yali_means, yali_stds = [], []
            nccl_means, nccl_stds = [], []

            for size in sizes:
                yali_stats = self.results.get(exec_mode, {}).get("yali", {}).get(dtype, {}).get("cuda-events", {}).get(size)
                nccl_stats = self.results.get(exec_mode, {}).get("nccl", {}).get(dtype, {}).get("cuda-events", {}).get(size)

                yali_means.append(yali_stats.mean if yali_stats else 0)
                yali_stds.append(yali_stats.stddev if yali_stats else 0)
                nccl_means.append(nccl_stats.mean if nccl_stats else 0)
                nccl_stds.append(nccl_stats.stddev if nccl_stats else 0)

            ax.bar(x - width/2, yali_means, width, yerr=yali_stds, label='YALI',
                   color=COLORS["yali_single"], capsize=3, edgecolor='white')
            ax.bar(x + width/2, nccl_means, width, yerr=nccl_stds, label='NCCL',
                   color=COLORS["nccl_single"], capsize=3, edgecolor='white')

        ax.set_xlabel('Message Size', fontsize=12)
        ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
        ax.set_title(f'Bandwidth with Std Dev Error Bars ({dtype.upper()})', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "graphs" / dtype / "variance_errorbar.png", dpi=150)
        plt.close()

    def _generate_variance_coefficient_graph(self, dtype: str):
        """Generate coefficient of variation graph."""
        sizes = self.config.sizes

        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(sizes))
        width = 0.35

        yali_cv, nccl_cv = [], []
        for size in sizes:
            yali_stats = self.results.get("single", {}).get("yali", {}).get(dtype, {}).get("cuda-events", {}).get(size)
            nccl_stats = self.results.get("single", {}).get("nccl", {}).get(dtype, {}).get("cuda-events", {}).get(size)
            yali_cv.append(yali_stats.cv_percent() if yali_stats else 0)
            nccl_cv.append(nccl_stats.cv_percent() if nccl_stats else 0)

        ax.bar(x - width/2, yali_cv, width, label='YALI', color=COLORS["yali_single"])
        ax.bar(x + width/2, nccl_cv, width, label='NCCL', color=COLORS["nccl_single"])

        ax.set_xlabel('Message Size')
        ax.set_ylabel('Coefficient of Variation (%)')
        ax.set_title(f'Measurement Stability - CV% ({dtype.upper()})\nLower = More Stable', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "graphs" / dtype / "variance_coefficient.png", dpi=150)
        plt.close()

    def _generate_min_max_spread_graph(self, dtype: str):
        """Generate min-max spread graph."""
        sizes = self.config.sizes

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(sizes))

        for lib, color, offset in [("yali", COLORS["yali_single"], -0.15),
                                    ("nccl", COLORS["nccl_single"], 0.15)]:
            means, mins, maxs = [], [], []
            for size in sizes:
                stats = self.results.get("single", {}).get(lib, {}).get(dtype, {}).get("cuda-events", {}).get(size)
                means.append(stats.mean if stats else 0)
                mins.append(stats.min_val if stats else 0)
                maxs.append(stats.max_val if stats else 0)

            # Plot mean as line
            ax.plot(x + offset, means, 'o-', color=color, label=f'{lib.upper()} Mean', markersize=8)
            # Plot min-max as error bars
            yerr = [[m - mi for m, mi in zip(means, mins)], [ma - m for m, ma in zip(means, maxs)]]
            ax.errorbar(x + offset, means, yerr=yerr, fmt='none', color=color, capsize=5, capthick=2, alpha=0.5)

        ax.set_xlabel('Message Size', fontsize=12)
        ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
        ax.set_title(f'Min-Max Spread ({dtype.upper()})', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, rotation=45, ha='right')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "graphs" / dtype / "min_max_spread.png", dpi=150)
        plt.close()

    def _generate_latency_boxplot(self, dtype: str):
        """Generate latency box plots."""
        sizes = self.config.sizes

        fig, ax = plt.subplots(figsize=(14, 6))

        positions = []
        data_yali = []
        data_nccl = []

        for i, size in enumerate(sizes):
            yali_stats = self.results.get("single", {}).get("yali", {}).get(dtype, {}).get("cuda-events", {}).get(size)
            nccl_stats = self.results.get("single", {}).get("nccl", {}).get(dtype, {}).get("cuda-events", {}).get(size)

            if yali_stats and yali_stats.samples:
                data_yali.append(yali_stats.samples)
                positions.append(i * 3)
            if nccl_stats and nccl_stats.samples:
                data_nccl.append(nccl_stats.samples)

        if data_yali:
            bp1 = ax.boxplot(data_yali, positions=[i*3 for i in range(len(data_yali))],
                            widths=0.8, patch_artist=True)
            for patch in bp1['boxes']:
                patch.set_facecolor(COLORS["yali_single"])

        if data_nccl:
            bp2 = ax.boxplot(data_nccl, positions=[i*3 + 1 for i in range(len(data_nccl))],
                            widths=0.8, patch_artist=True)
            for patch in bp2['boxes']:
                patch.set_facecolor(COLORS["nccl_single"])

        ax.set_xlabel('Message Size', fontsize=12)
        ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
        ax.set_title(f'Bandwidth Distribution Box Plot ({dtype.upper()})', fontsize=14, fontweight='bold')
        ax.set_xticks([i*3 + 0.5 for i in range(len(sizes))])
        ax.set_xticklabels(sizes, rotation=45, ha='right')
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]] if data_yali and data_nccl else [],
                  ['YALI', 'NCCL'], loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "graphs" / dtype / "latency_boxplot.png", dpi=150)
        plt.close()

    def _generate_latency_percentiles_graph(self, dtype: str):
        """Generate percentile comparison graph."""
        sizes = self.config.sizes

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(sizes))

        for lib, color, linestyle in [("yali", COLORS["yali_single"], "-"),
                                       ("nccl", COLORS["nccl_single"], "--")]:
            p50, p90, p95, p99 = [], [], [], []
            for size in sizes:
                stats = self.results.get("single", {}).get(lib, {}).get(dtype, {}).get("cuda-events", {}).get(size)
                p50.append(stats.p50 if stats else 0)
                p90.append(stats.p90 if stats else 0)
                p95.append(stats.p95 if stats else 0)
                p99.append(stats.p99 if stats else 0)

            ax.plot(x, p50, f'{linestyle}o', color=color, label=f'{lib.upper()} P50', alpha=1.0)
            ax.plot(x, p95, f'{linestyle}s', color=color, label=f'{lib.upper()} P95', alpha=0.7)
            ax.plot(x, p99, f'{linestyle}^', color=color, label=f'{lib.upper()} P99', alpha=0.5)

        ax.set_xlabel('Message Size', fontsize=12)
        ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
        ax.set_title(f'Percentile Comparison ({dtype.upper()})', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, rotation=45, ha='right')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "graphs" / dtype / "latency_percentiles.png", dpi=150)
        plt.close()

    def _generate_latency_cdf_graph(self, dtype: str):
        """Generate CDF graph for selected sizes."""
        key_sizes = ["16M", "128M", "2G"]
        key_sizes = [s for s in key_sizes if s in self.config.sizes]

        fig, axes = plt.subplots(1, len(key_sizes), figsize=(5*len(key_sizes), 4))
        if len(key_sizes) == 1:
            axes = [axes]

        for ax, size in zip(axes, key_sizes):
            for lib, color in [("yali", COLORS["yali_single"]), ("nccl", COLORS["nccl_single"])]:
                stats = self.results.get("single", {}).get(lib, {}).get(dtype, {}).get("cuda-events", {}).get(size)
                if stats and stats.samples:
                    sorted_data = np.sort(stats.samples)
                    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                    ax.plot(sorted_data, cdf, color=color, label=lib.upper(), linewidth=2)

            ax.set_xlabel('Bandwidth (GB/s)')
            ax.set_ylabel('CDF')
            ax.set_title(f'{size}')
            ax.legend()
            ax.grid(alpha=0.3)

        plt.suptitle(f'Cumulative Distribution Function ({dtype.upper()})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "graphs" / dtype / "latency_cdf.png", dpi=150)
        plt.close()

    def _generate_executive_summary_graph(self):
        """Generate executive summary graph."""
        fig, ax = plt.subplots(figsize=(12, 6))

        dtypes = list(self.config.dtypes.keys())
        x = np.arange(len(dtypes))
        width = 0.2

        bars_data = []
        for i, (exec_mode, lib) in enumerate([("single", "yali"), ("single", "nccl"),
                                               ("mpi", "yali"), ("mpi", "nccl")]):
            if exec_mode not in self.config.exec_modes:
                continue
            peaks = []
            for dtype in dtypes:
                max_bw = 0
                for timing in self.config.timing_modes:
                    for size in self.config.sizes:
                        stats = self.results.get(exec_mode, {}).get(lib, {}).get(dtype, {}).get(timing, {}).get(size)
                        if stats and stats.mean > max_bw:
                            max_bw = stats.mean
                peaks.append(max_bw)
            bars_data.append((f"{lib.upper()} {exec_mode.title()}", peaks, COLORS[f"{lib}_{exec_mode}"]))

        for i, (label, peaks, color) in enumerate(bars_data):
            offset = (i - len(bars_data)/2 + 0.5) * width
            ax.bar(x + offset, peaks, width, label=label, color=color, edgecolor='white')

        ax.axhline(y=self.d2d_bw, color='gray', linestyle='--', alpha=0.7, label=f'SoL')

        ax.set_xlabel('Data Type', fontsize=12)
        ax.set_ylabel('Peak Bandwidth (GB/s)', fontsize=12)
        ax.set_title('Executive Summary: Peak Performance by Dtype', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in dtypes])
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "graphs" / "executive_summary.png", dpi=150)
        plt.close()

    def _generate_speedup_heatmap(self):
        """Generate speedup heatmap."""
        sizes = self.config.sizes
        dtypes = list(self.config.dtypes.keys())

        # Create heatmap data
        data = np.zeros((len(dtypes), len(sizes)))

        for i, dtype in enumerate(dtypes):
            for j, size in enumerate(sizes):
                yali = self.results.get("single", {}).get("yali", {}).get(dtype, {}).get("cuda-events", {}).get(size)
                nccl = self.results.get("single", {}).get("nccl", {}).get(dtype, {}).get("cuda-events", {}).get(size)
                if yali and nccl and nccl.mean > 0:
                    data[i, j] = yali.mean / nccl.mean
                else:
                    data[i, j] = 1.0

        fig, ax = plt.subplots(figsize=(14, 4))
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.8, vmax=1.5)

        ax.set_xticks(np.arange(len(sizes)))
        ax.set_yticks(np.arange(len(dtypes)))
        ax.set_xticklabels(sizes, rotation=45, ha='right')
        ax.set_yticklabels([d.upper() for d in dtypes])

        # Add text annotations
        for i in range(len(dtypes)):
            for j in range(len(sizes)):
                text = ax.text(j, i, f'{data[i, j]:.2f}x', ha='center', va='center',
                              color='black' if 0.9 < data[i, j] < 1.3 else 'white', fontsize=8)

        ax.set_title('Speedup Heatmap (YALI/NCCL)', fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=ax, label='Speedup')

        plt.tight_layout()
        plt.savefig(self.output_dir / "graphs" / "speedup_heatmap.png", dpi=150)
        plt.close()

    # =========================================================================
    # Phase 6b: Profiler (nsys)
    # =========================================================================
    def run_profiled_benchmarks(self):
        """Run benchmarks with nsys profiling and extract kernel timings."""
        if not self.config.with_profiler:
            return

        self.progress.start_phase("Profiler")
        profiler_dir = self.output_dir / "profiler"
        profiler_dir.mkdir(exist_ok=True)

        # Profile key sizes only (to keep profiling fast)
        profile_sizes = ["1M", "64M", "256M"]
        bazel_bin = self.get_bazel_bin()
        env = {"CUDA_VISIBLE_DEVICES": "0,1"}

        self.profiler_results = {}

        for bench_type in ["yali", "nccl"]:
            self.profiler_results[bench_type] = {}
            binary = f"benchmark_{bench_type}"

            for size in profile_sizes:
                elements = SIZE_BYTES[size] // 4  # FP32 elements
                nsys_output = profiler_dir / f"{bench_type}_{size}"

                # Build command with nsys profile
                if bench_type == "yali":
                    args = [str(elements), str(self.config.calls), "0", "auto", "0", "cuda-events", "fp32"]
                else:
                    args = [str(elements), str(self.config.calls), "cuda-events", "fp32"]
                    env["LD_LIBRARY_PATH"] = f"third_party/nccl/build/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

                cmd = [
                    "nsys", "profile",
                    "--stats=true",
                    "--force-overwrite=true",
                    "-o", str(nsys_output),
                    f"{bazel_bin}/{binary}"
                ] + args

                result = self.run_cmd(cmd, env=env, timeout=300)

                # Export to SQLite
                nsys_rep = f"{nsys_output}.nsys-rep"
                sqlite_file = f"{nsys_output}.sqlite"

                if Path(nsys_rep).exists():
                    export_cmd = ["nsys", "export", "-t", "sqlite", "-o", sqlite_file, nsys_rep]
                    self.run_cmd(export_cmd, timeout=60)

                    # Extract kernel data
                    if Path(sqlite_file).exists():
                        kernel_data = self._extract_nsys_kernel_data(sqlite_file, bench_type)
                        self.profiler_results[bench_type][size] = kernel_data

        # Generate profiler graphs
        self._generate_profiler_graphs()
        self.progress.end_phase()

    def _extract_nsys_kernel_data(self, sqlite_file: str, bench_type: str) -> Dict[str, Any]:
        """Extract kernel timing data from nsys SQLite export."""
        try:
            conn = sqlite3.connect(sqlite_file)
            cursor = conn.cursor()

            # Get kernel names and durations
            query = """
                SELECT
                    s.value as kernel_name,
                    k.start,
                    k.end,
                    (k.end - k.start) as duration_ns
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON k.demangledName = s.id
                ORDER BY k.start
            """

            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return {}

            # Filter for relevant kernels
            relevant_kernels = []
            for name, start, end, duration_ns in rows:
                # Include YALI kernels (FlashKernel, _YaliKernel, yali::*)
                # Include NCCL kernels (nccl*, reduce)
                # Exclude seed/init kernels
                name_lower = name.lower()
                is_yali = bench_type == "yali" and (
                    "yali" in name_lower or
                    "flashkernel" in name_lower or
                    "_yalikernel" in name_lower
                )
                is_nccl = bench_type == "nccl" and (
                    "nccl" in name_lower or
                    "reduce" in name_lower
                )
                is_seed = "seed" in name_lower or "init" in name_lower

                if (is_yali or is_nccl) and not is_seed:
                    relevant_kernels.append({
                        "name": name,
                        "duration_us": duration_ns / 1000.0,
                        "start_ns": start,
                        "end_ns": end
                    })

            if not relevant_kernels:
                # Fallback: get all kernels
                relevant_kernels = [{
                    "name": name,
                    "duration_us": duration_ns / 1000.0,
                    "start_ns": start,
                    "end_ns": end
                } for name, start, end, duration_ns in rows]

            # Aggregate stats
            total_duration_us = sum(k["duration_us"] for k in relevant_kernels)
            kernel_count = len(relevant_kernels)
            avg_duration_us = total_duration_us / kernel_count if kernel_count > 0 else 0

            # Calculate wall clock time (first start to last end) - fair comparison
            first_start_ns = min(k["start_ns"] for k in relevant_kernels)
            last_end_ns = max(k["end_ns"] for k in relevant_kernels)
            wall_clock_us = (last_end_ns - first_start_ns) / 1000.0

            # Group by kernel name
            kernel_breakdown = {}
            for k in relevant_kernels:
                short_name = k["name"].split("<")[0].split("::")[-1][:30]
                if short_name not in kernel_breakdown:
                    kernel_breakdown[short_name] = {"count": 0, "total_us": 0}
                kernel_breakdown[short_name]["count"] += 1
                kernel_breakdown[short_name]["total_us"] += k["duration_us"]

            return {
                "total_duration_us": total_duration_us,
                "wall_clock_us": wall_clock_us,  # Fair comparison metric
                "kernel_count": kernel_count,
                "avg_duration_us": avg_duration_us,
                "breakdown": kernel_breakdown,
                "kernels": relevant_kernels[:100]  # Keep first 100 for detail
            }

        except Exception as e:
            print(f"Warning: Failed to extract nsys data from {sqlite_file}: {e}")
            return {}

    def _generate_profiler_graphs(self):
        """Generate profiler comparison graphs showing fair YALI vs NCCL comparisons."""
        if not hasattr(self, 'profiler_results') or not self.profiler_results:
            return

        profiler_dir = self.output_dir / "profiler"
        sizes = list(self.profiler_results.get("yali", {}).keys())
        if not sizes:
            return

        # Helper to parse size string to bytes
        def size_to_bytes(size_str: str) -> int:
            size_str = size_str.upper()
            multipliers = {'K': 1024, 'M': 1024**2, 'G': 1024**3}
            for suffix, mult in multipliers.items():
                if suffix in size_str:
                    return int(float(size_str.replace(suffix, '')) * mult)
            return int(size_str)

        # Graph 1: Effective Bandwidth (fair metric: bytes / total_kernel_time)
        # This shows actual throughput regardless of kernel strategy (many small vs few large kernels)
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(sizes))
        width = 0.35

        yali_bw = []
        nccl_bw = []
        for size in sizes:
            size_bytes = size_to_bytes(size)
            # Use wall_clock_us (first start to last end) for fair comparison
            # This accounts for overlapping/parallel kernel execution
            yali_wall_us = self.profiler_results.get("yali", {}).get(size, {}).get("wall_clock_us", 0)
            nccl_wall_us = self.profiler_results.get("nccl", {}).get(size, {}).get("wall_clock_us", 0)
            # Effective bandwidth = bytes / wall_clock_time (GB/s)
            yali_bw.append((size_bytes / 1e9) / (yali_wall_us / 1e6) if yali_wall_us > 0 else 0)
            nccl_bw.append((size_bytes / 1e9) / (nccl_wall_us / 1e6) if nccl_wall_us > 0 else 0)

        bars1 = ax.bar(x - width/2, yali_bw, width, label='YALI', color=COLORS["yali_single"], edgecolor='white')
        bars2 = ax.bar(x + width/2, nccl_bw, width, label='NCCL', color=COLORS["nccl_single"], edgecolor='white')

        # Add value labels and speedup
        for i, (yb, nb) in enumerate(zip(yali_bw, nccl_bw)):
            ax.text(x[i] - width/2, yb + 0.5, f'{yb:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(x[i] + width/2, nb + 0.5, f'{nb:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            if nb > 0:
                speedup = yb / nb
                color = 'green' if speedup > 1 else 'red'
                ax.text(x[i], max(yb, nb) + 3, f'{speedup:.2f}x', ha='center', va='bottom',
                       fontsize=10, fontweight='bold', color=color)

        ax.set_xlabel('Message Size', fontsize=12)
        ax.set_ylabel('Effective Kernel Bandwidth (GB/s)', fontsize=12)
        ax.set_title('Effective Kernel Bandwidth: YALI vs NCCL\n(bytes ÷ wall clock time = fair comparison)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sizes)
        ax.legend(loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(profiler_dir / "effective_bandwidth.png", dpi=150)
        plt.close()

        # Graph 2: Per-kernel Duration (only for sizes with comparable kernel counts)
        # Filter to sizes where kernel count ratio is <= 2x (fair apples-to-apples comparison)
        comparable_sizes = []
        for size in sizes:
            yali_count = self.profiler_results.get("yali", {}).get(size, {}).get("kernel_count", 0)
            nccl_count = self.profiler_results.get("nccl", {}).get(size, {}).get("kernel_count", 0)
            if yali_count > 0 and nccl_count > 0:
                ratio = max(yali_count, nccl_count) / min(yali_count, nccl_count)
                if ratio <= 2.0:
                    comparable_sizes.append(size)

        if comparable_sizes:
            fig, axes = plt.subplots(1, len(comparable_sizes), figsize=(5 * len(comparable_sizes), 5))
            if len(comparable_sizes) == 1:
                axes = [axes]

            for idx, size in enumerate(comparable_sizes):
                ax = axes[idx]
                yali_data = self.profiler_results.get("yali", {}).get(size, {})
                nccl_data = self.profiler_results.get("nccl", {}).get(size, {})

                yali_avg = yali_data.get("avg_duration_us", 0)
                nccl_avg = nccl_data.get("avg_duration_us", 0)
                yali_count = yali_data.get("kernel_count", 0)
                nccl_count = nccl_data.get("kernel_count", 0)

                bars = ax.bar(["YALI", "NCCL"], [yali_avg, nccl_avg],
                             color=[COLORS["yali_single"], COLORS["nccl_single"]], edgecolor='white')

                for bar, val in zip(bars, [yali_avg, nccl_avg]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{val:.1f} µs', ha='center', va='bottom', fontsize=10, fontweight='bold')

                if nccl_avg > 0 and yali_avg > 0:
                    speedup = nccl_avg / yali_avg
                    title = f'{size}\nYALI: {yali_count} kernels, NCCL: {nccl_count} kernels'
                    if speedup > 1:
                        title += f'\n({speedup:.2f}x faster per kernel)'
                    ax.set_title(title, fontsize=11)
                else:
                    ax.set_title(f'{size}', fontsize=12)

                ax.set_ylabel('Avg Kernel Duration (µs)')
                ax.grid(axis='y', alpha=0.3)

            plt.suptitle('Per-Kernel Duration: YALI vs NCCL\n(only sizes with comparable kernel counts shown)',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(profiler_dir / "kernel_duration_comparison.png", dpi=150)
            plt.close()
        else:
            # No comparable sizes, still generate a note
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5,
                   'No comparable kernel counts found.\n\nYALI uses different kernel strategies for different sizes:\n'
                   '- Small messages: Flash kernel (few large kernels)\n'
                   '- Large messages: Stream kernel (many small kernels)\n\n'
                   'See effective_bandwidth.png for fair comparison.',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.axis('off')
            plt.savefig(profiler_dir / "kernel_duration_comparison.png", dpi=150)
            plt.close()

        # Save profiler data as JSON with effective bandwidth (using wall clock time)
        profiler_summary = {
            "sizes": sizes,
            "yali": {s: {
                "wall_clock_ms": self.profiler_results.get("yali", {}).get(s, {}).get("wall_clock_us", 0) / 1000,
                "sum_kernel_ms": self.profiler_results.get("yali", {}).get(s, {}).get("total_duration_us", 0) / 1000,
                "avg_us": self.profiler_results.get("yali", {}).get(s, {}).get("avg_duration_us", 0),
                "kernel_count": self.profiler_results.get("yali", {}).get(s, {}).get("kernel_count", 0),
                "effective_bw_gbps": yali_bw[i] if i < len(yali_bw) else 0
            } for i, s in enumerate(sizes)},
            "nccl": {s: {
                "wall_clock_ms": self.profiler_results.get("nccl", {}).get(s, {}).get("wall_clock_us", 0) / 1000,
                "sum_kernel_ms": self.profiler_results.get("nccl", {}).get(s, {}).get("total_duration_us", 0) / 1000,
                "avg_us": self.profiler_results.get("nccl", {}).get(s, {}).get("avg_duration_us", 0),
                "kernel_count": self.profiler_results.get("nccl", {}).get(s, {}).get("kernel_count", 0),
                "effective_bw_gbps": nccl_bw[i] if i < len(nccl_bw) else 0
            } for i, s in enumerate(sizes)}
        }

        with open(profiler_dir / "profiler_summary.json", "w") as f:
            json.dump(profiler_summary, f, indent=2)

    # =========================================================================
    # Phase 7: Summary Report
    # =========================================================================
    def generate_summary(self, sys_info: Dict, hw_baseline: Dict, example_results: Dict):
        """Generate comprehensive summary markdown report."""
        self.progress.start_phase("Summary")

        lines = []

        # Header
        lines.append("# YALI vs NCCL AllReduce Performance Comparison")
        lines.append("")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if sys_info.get("gpus"):
            gpu = sys_info["gpus"][0]
            lines.append(f"**Platform:** 2x {gpu.get('name', 'NVIDIA GPU')} (NVLink)")
        lines.append(f"**Mode:** {self.config.mode.title()} | "
                    f"Dtypes: {', '.join(d.upper() for d in self.config.dtypes)} | "
                    f"Sizes: {len(self.config.sizes)} | Runs: {self.config.runs}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Executive Summary Graph
        lines.append("## Executive Summary")
        lines.append("")
        lines.append("![Executive Summary](graphs/executive_summary.png)")
        lines.append("")

        # Executive Summary Table
        summary_table = []
        for dtype in self.config.dtypes:
            row = [dtype.upper()]
            for exec_mode in self.config.exec_modes:
                yali_peak = max((self.results.get(exec_mode, {}).get("yali", {}).get(dtype, {}).get("cuda-events", {}).get(s, BenchStats(0,0,0,0,[])).mean for s in self.config.sizes), default=0)
                nccl_peak = max((self.results.get(exec_mode, {}).get("nccl", {}).get(dtype, {}).get("cuda-events", {}).get(s, BenchStats(0,0,0,0,[])).mean for s in self.config.sizes), default=0)
                speedup = yali_peak / nccl_peak if nccl_peak > 0 else 0
                row.extend([f"{yali_peak:.1f}", f"{nccl_peak:.1f}", format_speedup(speedup)])
            summary_table.append(row)

        headers = ["Dtype"]
        for exec_mode in self.config.exec_modes:
            headers.extend([f"{exec_mode.title()} YALI", f"{exec_mode.title()} NCCL", "Speedup"])

        lines.append("```")
        lines.append(tabulate(summary_table, headers=headers, tablefmt="pretty"))
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Hardware Baseline
        lines.append("## Hardware Baseline")
        lines.append("")
        hw_table = [
            ["nvbandwidth D2D (unidir)", f"{hw_baseline.get('d2d_unidirectional', 0):.2f} GB/s"],
            ["nvbandwidth D2D (bidir)", f"{hw_baseline.get('d2d_bidirectional', 0):.2f} GB/s"],
            ["NVLink", sys_info.get("nvlink", {}).get("version", "unknown")],
        ]
        lines.append("```")
        lines.append(tabulate(hw_table, headers=["Metric", "Value"], tablefmt="pretty"))
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Example Results
        lines.append("## Example Correctness")
        lines.append("")
        ex_table = [[name, "PASS" if passed else "FAIL"] for name, passed in example_results.items()]
        lines.append("```")
        lines.append(tabulate(ex_table, headers=["Example", "Status"], tablefmt="pretty"))
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Per-dtype detailed results
        for dtype in self.config.dtypes:
            lines.append(f"## {dtype.upper()} Results")
            lines.append("")

            # Graphs
            lines.append(f"### Bandwidth Comparison")
            lines.append(f"![Bandwidth {dtype.upper()}](graphs/{dtype}/bandwidth_comparison.png)")
            lines.append("")

            lines.append(f"### Speedup Analysis")
            lines.append(f"![Speedup {dtype.upper()}](graphs/{dtype}/speedup_by_mode.png)")
            lines.append("")

            lines.append(f"### Improvement Percentage")
            lines.append(f"![Improvement {dtype.upper()}](graphs/{dtype}/speedup_percentage.png)")
            lines.append("")

            if self.config.mode in ["full", "extensive"]:
                lines.append(f"### Timing Mode Comparison")
                lines.append(f"![Timing {dtype.upper()}](graphs/{dtype}/timing_mode_comparison.png)")
                lines.append("")

            # Tables per exec_mode and timing
            for exec_mode in self.config.exec_modes:
                for timing in self.config.timing_modes:
                    lines.append(f"### {exec_mode.title()} - {timing}")
                    lines.append("")

                    table_data = []
                    for size in self.config.sizes:
                        yali = self.results.get(exec_mode, {}).get("yali", {}).get(dtype, {}).get(timing, {}).get(size)
                        nccl = self.results.get(exec_mode, {}).get("nccl", {}).get(dtype, {}).get(timing, {}).get(size)

                        if yali and nccl:
                            speedup = yali.mean / nccl.mean if nccl.mean > 0 else 0
                            table_data.append([
                                bytes_to_human(SIZE_BYTES[size]),
                                yali.format(),
                                f"{self.sol_pct(yali.mean):.0f}%",
                                nccl.format(),
                                f"{self.sol_pct(nccl.mean):.0f}%",
                                format_speedup(speedup)
                            ])

                    if table_data:
                        lines.append("```")
                        lines.append(tabulate(table_data,
                                              headers=["Size", "YALI (GB/s)", "SoL%", "NCCL (GB/s)", "SoL%", "Speedup"],
                                              tablefmt="pretty"))
                        lines.append("```")
                    lines.append("")

            lines.append("---")
            lines.append("")

        # Extensive mode: Stability and Latency sections
        if self.config.mode == "extensive":
            lines.append("## Stability Analysis (Extensive Mode)")
            lines.append("")

            for dtype in self.config.dtypes:
                lines.append(f"### {dtype.upper()} Stability")
                lines.append("")
                lines.append(f"![Variance {dtype.upper()}](graphs/{dtype}/variance_errorbar.png)")
                lines.append("")
                lines.append(f"![CV {dtype.upper()}](graphs/{dtype}/variance_coefficient.png)")
                lines.append("")
                lines.append(f"![MinMax {dtype.upper()}](graphs/{dtype}/min_max_spread.png)")
                lines.append("")

            lines.append("---")
            lines.append("")

            lines.append("## Latency Analysis (Extensive Mode)")
            lines.append("")

            for dtype in self.config.dtypes:
                lines.append(f"### {dtype.upper()} Latency Distribution")
                lines.append("")
                lines.append(f"![Boxplot {dtype.upper()}](graphs/{dtype}/latency_boxplot.png)")
                lines.append("")
                lines.append(f"![Percentiles {dtype.upper()}](graphs/{dtype}/latency_percentiles.png)")
                lines.append("")
                lines.append(f"![CDF {dtype.upper()}](graphs/{dtype}/latency_cdf.png)")
                lines.append("")

                # Percentile table
                lines.append(f"#### {dtype.upper()} Percentiles Table")
                lines.append("")
                pct_table = []
                for size in self.config.sizes:
                    yali = self.results.get("single", {}).get("yali", {}).get(dtype, {}).get("cuda-events", {}).get(size)
                    nccl = self.results.get("single", {}).get("nccl", {}).get(dtype, {}).get("cuda-events", {}).get(size)
                    if yali and nccl:
                        pct_table.append([
                            size,
                            f"{yali.p50:.1f}", f"{yali.p95:.1f}", f"{yali.p99:.1f}",
                            f"{nccl.p50:.1f}", f"{nccl.p95:.1f}", f"{nccl.p99:.1f}",
                        ])

                if pct_table:
                    lines.append("```")
                    lines.append(tabulate(pct_table,
                                          headers=["Size", "YALI P50", "YALI P95", "YALI P99",
                                                   "NCCL P50", "NCCL P95", "NCCL P99"],
                                          tablefmt="pretty"))
                    lines.append("```")
                lines.append("")

            lines.append("---")
            lines.append("")

        # Speedup Heatmap
        if self.config.mode in ["full", "extensive"]:
            lines.append("## Speedup Heatmap")
            lines.append("")
            lines.append("![Speedup Heatmap](graphs/speedup_heatmap.png)")
            lines.append("")
            lines.append("---")
            lines.append("")

        # Profiler Results (if enabled)
        if self.config.with_profiler and hasattr(self, 'profiler_results') and self.profiler_results:
            lines.append("## Profiler Results (nsys)")
            lines.append("")
            lines.append("Kernel-level timing captured via NVIDIA Nsight Systems.")
            lines.append("")

            # Effective Bandwidth graph - the fair comparison metric
            lines.append("### Effective Kernel Bandwidth")
            lines.append("")
            lines.append("Fair comparison metric: `bytes ÷ wall_clock_time = GB/s`")
            lines.append("")
            lines.append("*Wall clock = first kernel start to last kernel end (accounts for overlapping kernels)*")
            lines.append("")
            lines.append("![Effective Bandwidth](profiler/effective_bandwidth.png)")
            lines.append("")

            # Kernel Duration Comparison graph (only for comparable kernel counts)
            lines.append("### Per-Kernel Duration")
            lines.append("")
            lines.append("*Only shown for message sizes with comparable kernel counts (≤2x ratio)*")
            lines.append("")
            lines.append("![Kernel Duration](profiler/kernel_duration_comparison.png)")
            lines.append("")

            # Profiler data table with effective bandwidth
            lines.append("### Profiler Summary")
            lines.append("")

            # Helper to parse size string to bytes
            def size_to_bytes(size_str: str) -> int:
                size_str = size_str.upper()
                multipliers = {'K': 1024, 'M': 1024**2, 'G': 1024**3}
                for suffix, mult in multipliers.items():
                    if suffix in size_str:
                        return int(float(size_str.replace(suffix, '')) * mult)
                return int(size_str)

            profiler_table = []
            sizes = list(self.profiler_results.get("yali", {}).keys())
            for size in sizes:
                yali_data = self.profiler_results.get("yali", {}).get(size, {})
                nccl_data = self.profiler_results.get("nccl", {}).get(size, {})

                # Use wall_clock_us for fair comparison (accounts for parallel kernels)
                yali_wall_us = yali_data.get("wall_clock_us", 0)
                nccl_wall_us = nccl_data.get("wall_clock_us", 0)
                yali_count = yali_data.get("kernel_count", 0)
                nccl_count = nccl_data.get("kernel_count", 0)

                # Calculate effective bandwidth (GB/s) using wall clock time
                size_bytes = size_to_bytes(size)
                yali_bw = (size_bytes / 1e9) / (yali_wall_us / 1e6) if yali_wall_us > 0 else 0
                nccl_bw = (size_bytes / 1e9) / (nccl_wall_us / 1e6) if nccl_wall_us > 0 else 0

                # Speedup based on effective bandwidth (fair comparison)
                speedup = yali_bw / nccl_bw if nccl_bw > 0 else 0

                profiler_table.append([
                    size,
                    f"{yali_bw:.1f} GB/s ({yali_count})",
                    f"{nccl_bw:.1f} GB/s ({nccl_count})",
                    format_speedup(speedup)
                ])

            if profiler_table:
                lines.append("```")
                lines.append(tabulate(profiler_table,
                                      headers=["Size", "YALI BW (kernels)", "NCCL BW (kernels)", "Speedup"],
                                      tablefmt="pretty"))
                lines.append("```")
            lines.append("")
            lines.append("---")
            lines.append("")

        # Reproducibility
        lines.append("## Reproducibility")
        lines.append("")
        lines.append("```bash")
        cmd_parts = ["python scripts/sweep.py"]
        if self.config.mode == "quick":
            cmd_parts.append("--quick")
        elif self.config.mode == "standard":
            cmd_parts.append("--standard")
        elif self.config.mode == "extensive":
            cmd_parts.append("--extensive")
        if self.config.with_profiler:
            cmd_parts.append("--profiler")
        if "mpi" not in self.config.exec_modes:
            cmd_parts.append("--single-only")
        elif "single" not in self.config.exec_modes:
            cmd_parts.append("--mpi-only")
        lines.append(" ".join(cmd_parts))
        lines.append("```")
        lines.append("")

        # Write report
        report_file = self.output_dir / "summary.md"
        with open(report_file, "w") as f:
            f.write("\n".join(lines))

        self.progress.end_phase()

        # Print final summary to console
        self._print_final_summary()

    def _print_final_summary(self):
        """Print final summary to console."""
        print("\n" + "=" * 70)
        print("  RESULTS SUMMARY")
        print("=" * 70)

        # Get peak values
        for dtype in list(self.config.dtypes.keys())[:1]:  # Just FP32 for console
            print(f"\n  {dtype.upper()}:")
            for exec_mode in self.config.exec_modes:
                yali_peak = max((self.results.get(exec_mode, {}).get("yali", {}).get(dtype, {}).get("cuda-events", {}).get(s, BenchStats(0,0,0,0,[])).mean for s in self.config.sizes), default=0)
                nccl_peak = max((self.results.get(exec_mode, {}).get("nccl", {}).get(dtype, {}).get("cuda-events", {}).get(s, BenchStats(0,0,0,0,[])).mean for s in self.config.sizes), default=0)
                speedup = yali_peak / nccl_peak if nccl_peak > 0 else 0
                print(f"    {exec_mode.title():8} YALI: {yali_peak:5.1f} GB/s  NCCL: {nccl_peak:5.1f} GB/s  {format_speedup(speedup)}")

        print(f"\n  Report: {self.output_dir}/summary.md")
        print("=" * 70)

    def _update_index(self):
        """Update/create index.md at the date level linking to all runs."""
        date_dir = self.output_dir.parent
        index_file = date_dir / "index.md"

        # Find all run folders in this date directory
        runs = []
        for run_dir in sorted(date_dir.iterdir()):
            if run_dir.is_dir() and (run_dir / "summary.md").exists():
                # Extract run info from folder name (e.g., "120750-quick")
                folder_name = run_dir.name
                parts = folder_name.split("-", 1)
                time_str = parts[0]
                mode = parts[1] if len(parts) > 1 else "full"

                # Read summary to get platform and results
                with open(run_dir / "summary.md") as f:
                    content = f.read()

                platform = ""
                single_speedup = ""
                mpi_speedup = ""

                for line in content.split("\n"):
                    if "Platform:" in line:
                        platform = line.split("Platform:")[1].strip().rstrip("*")
                    # Look for speedup in executive summary table
                    if "| FP32" in line and "|" in line:
                        parts_line = [p.strip() for p in line.split("|")]
                        if len(parts_line) >= 7:
                            # Format: | FP32 | single_yali | single_nccl | speedup | mpi_yali | mpi_nccl | speedup |
                            single_speedup = parts_line[4] if len(parts_line) > 4 else ""
                            mpi_speedup = parts_line[7] if len(parts_line) > 7 else ""

                runs.append({
                    "folder": folder_name,
                    "time": f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}",
                    "mode": mode.title(),
                    "platform": platform,
                    "single_speedup": single_speedup,
                    "mpi_speedup": mpi_speedup,
                })

        # Generate index.md
        lines = [
            f"# Benchmark Runs - {date_dir.name}",
            "",
        ]

        # Build table data for tabulate
        table_data = []
        for run in runs:
            single = run['single_speedup'] or "-"
            mpi = run['mpi_speedup'] or "-"
            report_link = f"[{run['folder']}/summary.md]({run['folder']}/summary.md)"
            table_data.append([run['time'], run['mode'], single, mpi, report_link])

        headers = ["Time", "Mode", "Single Speedup", "MPI Speedup", "Report"]
        lines.append(tabulate(table_data, headers=headers, tablefmt="pipe"))

        lines.extend([
            "",
            f"**Platform:** {runs[0]['platform'] if runs else 'Unknown'}",
            "",
            "---",
            "",
            "*Auto-generated index*"
        ])

        with open(index_file, "w") as f:
            f.write("\n".join(lines))

    # =========================================================================
    # Main Run
    # =========================================================================
    def run(self) -> bool:
        """Run the complete benchmark sweep."""
        self.progress.start()

        # Phase 1: Build
        if not self.build_targets():
            print("ERROR: Build failed")
            return False

        # Phase 2: System Info
        sys_info = self.collect_system_info()

        # Phase 3: Hardware Baseline
        hw_baseline = self.run_nvbandwidth()

        # Phase 4: Examples
        example_results = self.run_examples()

        # Phase 5: All Benchmarks
        self.run_all_benchmarks()

        # Phase 5b: Profiler (if enabled)
        self.run_profiled_benchmarks()

        # Phase 6: Graphs
        self.generate_graphs()

        # Phase 7: Summary
        self.generate_summary(sys_info, hw_baseline, example_results)

        # Update index.md at date level
        self._update_index()

        self.progress.finish()
        return True


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="YALI vs NCCL Comprehensive Benchmark Sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
    --quick      Quick sanity check (~2 min): FP32 only, 5 sizes, cuda-events
    --standard   All sizes coverage (~8 min): All dtypes, 11 sizes, cuda-events
    (default)    Full benchmark (~10 min): Key sizes (16M/128M), all timing modes
    --extensive  Statistical rigor (~30 min): Key sizes, all timing modes, 10 runs

Examples:
    %(prog)s --quick              # Quick sanity check
    %(prog)s --standard           # All sizes, cuda-events only
    %(prog)s                      # Key sizes, all timing modes
    %(prog)s --extensive          # Statistical analysis with 10 runs
    %(prog)s --single-only        # Skip MPI benchmarks
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--quick", action="store_true", help="Quick mode (~2 min)")
    mode_group.add_argument("--standard", action="store_true", help="Standard mode (~8 min)")
    mode_group.add_argument("--extensive", action="store_true", help="Extensive mode (~30 min)")

    # Execution mode
    exec_group = parser.add_mutually_exclusive_group()
    exec_group.add_argument("--single-only", action="store_true", help="Skip MPI benchmarks")
    exec_group.add_argument("--mpi-only", action="store_true", help="Skip single-process benchmarks")

    # Fine-grained control
    parser.add_argument("--runs", type=int, help="Override number of runs per size")
    parser.add_argument("--calls", type=int, default=20, help="Calls per run (default: 20)")
    parser.add_argument("--sizes", nargs="+", help="Custom sizes (e.g., 4K 16M 2G)")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--profiler", action="store_true", help="Enable nsys profiler (requires nsys)")

    args = parser.parse_args()

    # Determine config
    if args.quick:
        config = SweepConfig.quick()
    elif args.standard:
        config = SweepConfig.standard()
    elif args.extensive:
        config = SweepConfig.extensive()
    else:
        config = SweepConfig.full()

    # Override exec modes
    if args.single_only:
        config.exec_modes = ["single"]
    elif args.mpi_only:
        config.exec_modes = ["mpi"]

    # Override other settings
    if args.runs:
        config.runs = args.runs
    if args.calls:
        config.calls = args.calls
    if args.sizes:
        config.sizes = args.sizes
    if args.profiler:
        config.with_profiler = True

    # Determine mode name for folder
    if args.quick:
        mode_name = "quick"
    elif args.standard:
        mode_name = "standard"
    elif args.extensive:
        mode_name = "extensive"
    else:
        mode_name = "full"

    if args.profiler:
        mode_name += "-profiler"

    # Output directory with mode in folder name
    if args.output:
        output_dir = Path(args.output)
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H%M%S")
        output_dir = Path(f"output/{date_str}/{time_str}-{mode_name}")

    # Change to project directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    os.chdir(project_dir)

    # Run sweep
    runner = SweepRunner(output_dir, config, verbose=not args.quiet)
    success = runner.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
