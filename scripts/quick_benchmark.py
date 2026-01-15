#!/usr/bin/env python3
"""
Quick YALI vs NCCL Benchmark

Runs FP32 benchmarks at standard sizes and prints a comparison table.
Supports both single-process (2 GPUs in one process) and MPI mode.

Features:
- Multiple runs per size for statistical reliability
- Reports mean, stddev, min, max
- Uses NCCL busBw formula for fair comparison

Usage:
    python scripts/quick_benchmark.py           # Single-process mode
    python scripts/quick_benchmark.py --mpi     # MPI mode (2 processes)
    python scripts/quick_benchmark.py --sizes 64M 256M  # Custom sizes
    python scripts/quick_benchmark.py --runs 5  # 5 runs per size for statistics

Requirements:
    - 2 GPUs with NVLink
    - bazel build //:benchmark_yali //:benchmark_nccl
    - For MPI: bazel build //:benchmark_yali_mpi //:benchmark_nccl_mpi
"""

import argparse
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, List, Tuple

# Standard benchmark sizes (element counts for fp32)
SIZE_PRESETS = {
    "4K": 1024,           # 4 KB
    "16K": 4096,          # 16 KB
    "64K": 16384,         # 64 KB
    "256K": 65536,        # 256 KB
    "1M": 262144,         # 1 MB
    "4M": 1048576,        # 4 MB
    "16M": 4194304,       # 16 MB
    "64M": 16777216,      # 64 MB
    "128M": 33554432,     # 128 MB
    "256M": 67108864,     # 256 MB
    "512M": 134217728,    # 512 MB
    "1G": 268435456,      # 1 GB
    "2G": 536870912,      # 2 GB
}

DEFAULT_SIZES = ["4K", "16M", "64M", "128M", "2G"]


@dataclass
class BenchStats:
    """Statistics from multiple benchmark runs."""
    mean: float
    stddev: float
    min_val: float
    max_val: float
    samples: List[float]

    @classmethod
    def from_samples(cls, samples: List[float]) -> Optional['BenchStats']:
        """Calculate statistics from list of samples."""
        if not samples:
            return None
        n = len(samples)
        mean = sum(samples) / n
        if n > 1:
            variance = sum((x - mean) ** 2 for x in samples) / (n - 1)
            stddev = math.sqrt(variance)
        else:
            stddev = 0.0
        return cls(
            mean=mean,
            stddev=stddev,
            min_val=min(samples),
            max_val=max(samples),
            samples=samples
        )

    def cv_percent(self) -> float:
        """Coefficient of variation as percentage."""
        if self.mean == 0:
            return 0.0
        return (self.stddev / self.mean) * 100


def get_bazel_bin() -> str:
    """Get bazel-bin directory path."""
    result = subprocess.run(
        ["bazel", "info", "bazel-bin"],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return result.stdout.strip()


def parse_size(size_str: str) -> int:
    """Parse size string like '64M' to element count."""
    size_str = size_str.upper()
    if size_str in SIZE_PRESETS:
        return SIZE_PRESETS[size_str]
    # Try parsing with suffix
    match = re.match(r"(\d+)([KMG])?", size_str)
    if match:
        num = int(match.group(1))
        suffix = match.group(2)
        multiplier = {"K": 1024, "M": 1024**2, "G": 1024**3}.get(suffix, 1)
        return num * multiplier // 4  # Convert bytes to float elements
    raise ValueError(f"Cannot parse size: {size_str}")


def run_single_benchmark(cmd: List[str], env: dict, name: str) -> Optional[float]:
    """Run a single benchmark and extract GB/s."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
        for line in result.stdout.split("\n"):
            if "GB/s" in line and name in line:
                match = re.search(r"([\d.]+)\s*GB/s", line)
                if match:
                    return float(match.group(1))
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  {name} error: {e}", file=sys.stderr)
    return None


def run_benchmark_with_stats(
    bazel_bin: str,
    elements: int,
    num_runs: int,
    calls_per_run: int,
    mpi: bool,
    bench_type: str  # "YALI" or "NCCL"
) -> Optional[BenchStats]:
    """Run benchmark multiple times and collect statistics."""

    if bench_type == "YALI":
        if mpi:
            cmd = [
                "mpirun", "-np", "2", "--allow-run-as-root", "--bind-to", "none",
                "-x", "CUDA_VISIBLE_DEVICES",
                f"{bazel_bin}/benchmark_yali_mpi",
                str(elements), str(calls_per_run), "cuda-events"
            ]
        else:
            cmd = [
                f"{bazel_bin}/benchmark_yali",
                str(elements), str(calls_per_run), "cuda-events"
            ]
    else:  # NCCL
        if mpi:
            cmd = [
                "mpirun", "-np", "2", "--allow-run-as-root", "--bind-to", "none",
                "-x", "CUDA_VISIBLE_DEVICES",
                "-x", "LD_LIBRARY_PATH",
                f"{bazel_bin}/benchmark_nccl_mpi",
                str(elements), str(calls_per_run), "cuda-events"
            ]
        else:
            cmd = [
                f"{bazel_bin}/benchmark_nccl",
                str(elements), str(calls_per_run), "cuda-events"
            ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    if bench_type == "NCCL":
        env["LD_LIBRARY_PATH"] = f"third_party/nccl/build/lib:{env.get('LD_LIBRARY_PATH', '')}"

    samples = []
    for _ in range(num_runs):
        gbps = run_single_benchmark(cmd, env, bench_type)
        if gbps is not None:
            samples.append(gbps)

    return BenchStats.from_samples(samples)


def format_stats(stats: Optional[BenchStats], show_stddev: bool = True) -> str:
    """Format statistics for display."""
    if stats is None:
        return "ERROR"
    if show_stddev and len(stats.samples) > 1:
        return f"{stats.mean:.1f}±{stats.stddev:.1f}"
    return f"{stats.mean:.2f}"


def main():
    parser = argparse.ArgumentParser(
        description="Quick YALI vs NCCL Benchmark with Statistical Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                           # Default: 5 sizes, 3 runs each
    %(prog)s --mpi                     # MPI mode
    %(prog)s --runs 5                  # 5 runs per size for better stats
    %(prog)s --sizes 64M 128M 256M     # Custom sizes
    %(prog)s --detailed                # Show min/max and individual samples
        """
    )
    parser.add_argument("--sizes", nargs="+", default=DEFAULT_SIZES,
                        help=f"Sizes to benchmark (default: {' '.join(DEFAULT_SIZES)})")
    parser.add_argument("--mpi", action="store_true",
                        help="Use MPI mode (2 processes, 1 GPU each)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of runs per size for statistics (default: 3)")
    parser.add_argument("--calls", type=int, default=20,
                        help="Number of iterations per run (default: 20)")
    parser.add_argument("--detailed", action="store_true",
                        help="Show detailed statistics (min, max, CV)")
    args = parser.parse_args()

    # Get bazel bin path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    os.chdir(project_dir)

    bazel_bin = get_bazel_bin()
    mode_str = "MPI (2 processes)" if args.mpi else "Single-process (2 GPUs)"

    # Check binaries exist
    yali_bin = f"{bazel_bin}/benchmark_yali_mpi" if args.mpi else f"{bazel_bin}/benchmark_yali"
    nccl_bin = f"{bazel_bin}/benchmark_nccl_mpi" if args.mpi else f"{bazel_bin}/benchmark_nccl"

    missing = []
    if not os.path.exists(yali_bin):
        missing.append(yali_bin)
    if not os.path.exists(nccl_bin):
        missing.append(nccl_bin)

    if missing:
        print("Missing binaries. Build with:")
        if args.mpi:
            print("  bazel build //:benchmark_yali_mpi //:benchmark_nccl_mpi")
        else:
            print("  bazel build //:benchmark_yali //:benchmark_nccl")
        sys.exit(1)

    # Header
    print("=" * 78)
    print(f"YALI vs NCCL AllReduce Benchmark (FP32, {mode_str})")
    print(f"Runs per size: {args.runs}, Calls per run: {args.calls}")
    print("=" * 78)
    print()

    if args.detailed:
        print(f"{'Size':>8} {'YALI GB/s':>14} {'NCCL GB/s':>14} {'Speedup':>10} {'YALI CV%':>9} {'NCCL CV%':>9}")
        print("-" * 68)
    else:
        print(f"{'Size':>8} {'YALI (GB/s)':>16} {'NCCL (GB/s)':>16} {'Speedup':>12}")
        print("-" * 56)

    results: List[Tuple[str, Optional[BenchStats], Optional[BenchStats]]] = []

    for size_str in args.sizes:
        try:
            elements = parse_size(size_str)
        except ValueError:
            print(f"  Skipping invalid size: {size_str}", file=sys.stderr)
            continue

        mb = elements * 4 / 1e6
        if mb >= 1000:
            size_label = f"{mb/1000:.1f}GB"
        elif mb >= 1:
            size_label = f"{mb:.0f}MB"
        else:
            size_label = f"{mb*1000:.0f}KB"

        # Run benchmarks
        yali_stats = run_benchmark_with_stats(
            bazel_bin, elements, args.runs, args.calls, args.mpi, "YALI")
        nccl_stats = run_benchmark_with_stats(
            bazel_bin, elements, args.runs, args.calls, args.mpi, "NCCL")

        results.append((size_label, yali_stats, nccl_stats))

        # Format output
        yali_str = format_stats(yali_stats, show_stddev=(args.runs > 1))
        nccl_str = format_stats(nccl_stats, show_stddev=(args.runs > 1))

        if yali_stats and nccl_stats:
            speedup = yali_stats.mean / nccl_stats.mean
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "-"

        if args.detailed:
            yali_cv = f"{yali_stats.cv_percent():.1f}%" if yali_stats else "-"
            nccl_cv = f"{nccl_stats.cv_percent():.1f}%" if nccl_stats else "-"
            print(f"{size_label:>8} {yali_str:>14} {nccl_str:>14} {speedup_str:>10} {yali_cv:>9} {nccl_cv:>9}")
        else:
            print(f"{size_label:>8} {yali_str:>16} {nccl_str:>16} {speedup_str:>12}")

    if args.detailed:
        print("-" * 68)
    else:
        print("-" * 56)

    # Summary
    valid_results = [(y, n) for _, y, n in results if y and n]
    if valid_results:
        avg_speedup = sum(y.mean/n.mean for y, n in valid_results) / len(valid_results)
        if args.detailed:
            print(f"{'Average':>8} {'-':>14} {'-':>14} {avg_speedup:.2f}x")
        else:
            print(f"{'Average':>8} {'-':>16} {'-':>16} {avg_speedup:.2f}x")

    print()
    print("Statistics: mean±stddev (from multiple runs)")
    print("Note: Using NCCL busBw formula for 2 GPUs")

    # Detailed summary if requested
    if args.detailed and valid_results:
        print()
        print("=" * 78)
        print("Detailed Statistics")
        print("=" * 78)
        for size_label, yali_stats, nccl_stats in results:
            if yali_stats and nccl_stats:
                print(f"\n{size_label}:")
                print(f"  YALI: mean={yali_stats.mean:.2f}, stddev={yali_stats.stddev:.2f}, "
                      f"min={yali_stats.min_val:.2f}, max={yali_stats.max_val:.2f}")
                print(f"        samples: {[f'{s:.2f}' for s in yali_stats.samples]}")
                print(f"  NCCL: mean={nccl_stats.mean:.2f}, stddev={nccl_stats.stddev:.2f}, "
                      f"min={nccl_stats.min_val:.2f}, max={nccl_stats.max_val:.2f}")
                print(f"        samples: {[f'{s:.2f}' for s in nccl_stats.samples]}")


if __name__ == "__main__":
    main()
