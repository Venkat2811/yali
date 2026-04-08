#!/usr/bin/env python3
"""
Shared runtime target selection for architecture-specific Bazel outputs.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class RuntimeTargets:
    gpu_name: str
    bazel_config: str
    nccl_tests_target: str
    nccl_tests_mpi_target: str
    nccl_tests_binary: str
    nccl_tests_mpi_binary: str
    nvbandwidth_target: str

    def bazel_build_flags(self) -> List[str]:
        return [self.bazel_config] if self.bazel_config else []


def detect_runtime_targets(project_dir: Path) -> RuntimeTargets:
    gpu_name = _detect_gpu_name(project_dir)

    if "H200" in gpu_name:
        return RuntimeTargets(
            gpu_name=gpu_name,
            bazel_config="--config=h200",
            nccl_tests_target="//:nccl_tests_bin_h200",
            nccl_tests_mpi_target="//:nccl_tests_mpi_bin_h200",
            nccl_tests_binary="all_reduce_perf_hopper",
            nccl_tests_mpi_binary="all_reduce_perf_mpi_hopper",
            nvbandwidth_target="//:nvbandwidth_bin_h200",
        )
    if "H100" in gpu_name:
        return RuntimeTargets(
            gpu_name=gpu_name,
            bazel_config="--config=h100",
            nccl_tests_target="//:nccl_tests_bin_h100",
            nccl_tests_mpi_target="//:nccl_tests_mpi_bin_h100",
            nccl_tests_binary="all_reduce_perf_hopper",
            nccl_tests_mpi_binary="all_reduce_perf_mpi_hopper",
            nvbandwidth_target="//:nvbandwidth_bin_h100",
        )
    if "B200" in gpu_name:
        return RuntimeTargets(
            gpu_name=gpu_name,
            bazel_config="--config=b200",
            nccl_tests_target="//:nccl_tests_bin",
            nccl_tests_mpi_target="//:nccl_tests_mpi_bin",
            nccl_tests_binary="all_reduce_perf",
            nccl_tests_mpi_binary="all_reduce_perf_mpi",
            nvbandwidth_target="//:nvbandwidth_bin",
        )

    return RuntimeTargets(
        gpu_name=gpu_name,
        bazel_config="",
        nccl_tests_target="//:nccl_tests_bin",
        nccl_tests_mpi_target="//:nccl_tests_mpi_bin",
        nccl_tests_binary="all_reduce_perf",
        nccl_tests_mpi_binary="all_reduce_perf_mpi",
        nvbandwidth_target="//:nvbandwidth_bin",
    )


def _detect_gpu_name(project_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            cwd=project_dir,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""

    if result.returncode != 0:
        return ""

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else ""
