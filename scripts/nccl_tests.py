#!/usr/bin/env python3
"""
Helpers for running nccl-tests all_reduce_perf as an alternate NCCL backend.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from runtime_targets import detect_runtime_targets


DTYPE_MAP = {
    "fp32": "float",
    "fp16": "half",
    "bf16": "bfloat16",
}


@dataclass(frozen=True)
class NcclTestsConfig:
    mode: str = "1proc-1thr"
    api: str = "host"
    warmup_iters: int = 1
    iters: int = 5

    def is_mpi(self) -> bool:
        return self.mode == "mpi"


def resolve_nccl_tests_binary(project_dir: Path, bazel_bin: str, config: NcclTestsConfig) -> Path:
    runtime = detect_runtime_targets(project_dir)
    binary_name = runtime.nccl_tests_mpi_binary if config.is_mpi() else runtime.nccl_tests_binary
    return Path(bazel_bin) / binary_name


def run_nccl_tests(
    project_dir: Path,
    bazel_bin: str,
    elements: int,
    dtype: str,
    config: NcclTestsConfig,
    env: Dict[str, str],
    timeout: int = 180,
) -> Optional[float]:
    binary = resolve_nccl_tests_binary(project_dir, bazel_bin, config)
    if not binary.exists():
        return None

    size_bytes = elements * _dtype_size(dtype)
    cmd = _build_cmd(binary, size_bytes, dtype, config, env)

    full_env = dict(env)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=full_env,
        cwd=project_dir,
        timeout=timeout,
    )
    return parse_nccl_tests_busbw(result.stdout)


def parse_nccl_tests_busbw(output: str) -> Optional[float]:
    match = re.search(r"# Avg bus bandwidth\s*:\s*([\d.]+)", output)
    if match:
        return float(match.group(1))
    return None


def _build_cmd(binary: Path, size_bytes: int, dtype: str, config: NcclTestsConfig, env: Dict[str, str]) -> List[str]:
    args = [
        str(binary),
        "-b", str(size_bytes),
        "-e", str(size_bytes),
        "-w", str(config.warmup_iters),
        "-n", str(config.iters),
        "-d", DTYPE_MAP[dtype],
    ]

    if config.mode == "1proc-1thr":
        args[1:1] = ["-g", "2"]
    elif config.mode == "1proc-2thr":
        args[1:1] = ["-t", "2", "-g", "1"]
    elif config.mode == "mpi":
        args[1:1] = ["-g", "1"]
    else:
        raise ValueError(f"Unsupported nccl-tests mode: {config.mode}")

    if config.api == "device":
        args.extend(["-D", "2", "-R", "2", "-z", "1"])
    elif config.api != "host":
        raise ValueError(f"Unsupported nccl-tests API mode: {config.api}")

    if config.mode != "mpi":
        return args

    exported_keys = ["CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"]
    exported_keys.extend(sorted(key for key in env if key.startswith("NCCL_")))
    mpirun_cmd = ["mpirun", "-np", "2", "--allow-run-as-root", "--bind-to", "none"]
    for key in exported_keys:
        if key in env:
            mpirun_cmd.extend(["-x", key])
    mpirun_cmd.extend(args)
    return mpirun_cmd


def _dtype_size(dtype: str) -> int:
    if dtype == "fp32":
        return 4
    if dtype in ("fp16", "bf16"):
        return 2
    raise ValueError(f"Unsupported dtype: {dtype}")
