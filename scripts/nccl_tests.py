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
    register_mode: str = "none"
    cudagraph_launches: int = 0
    cta_policy: Optional[int] = None
    device_impl: int = 0
    device_cta_count: int = 16
    blocking_mode: Optional[int] = None

    def is_mpi(self) -> bool:
        return self.mode == "mpi"

    def effective_register_mode(self) -> str:
        if self.api == "device" and self.register_mode == "none":
            return "symmetric"
        return self.register_mode

    def effective_device_impl(self) -> int:
        if self.api == "device" and self.device_impl == 0:
            return 2
        return self.device_impl

    def effective_blocking_mode(self) -> Optional[int]:
        if self.api == "device" and self.blocking_mode is None:
            return 1
        return self.blocking_mode

    def describe(self) -> str:
        parts = [self.mode, self.api]
        register_mode = self.effective_register_mode()
        if register_mode != "none":
            parts.append(f"reg={register_mode}")
        if self.cudagraph_launches:
            parts.append(f"graph={self.cudagraph_launches}")
        if self.cta_policy is not None:
            parts.append(f"cta={self.cta_policy}")
        if self.effective_device_impl():
            parts.append(f"devimpl={self.effective_device_impl()}")
        if self.device_cta_count != 16:
            parts.append(f"devctas={self.device_cta_count}")
        blocking_mode = self.effective_blocking_mode()
        if blocking_mode is not None:
            parts.append(f"blocking={blocking_mode}")
        return ", ".join(parts)


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

    if config.api not in ("host", "device"):
        raise ValueError(f"Unsupported nccl-tests API mode: {config.api}")

    register_mode = config.effective_register_mode()
    register_value = {"none": 0, "local": 1, "symmetric": 2}.get(register_mode)
    if register_value is None:
        raise ValueError(f"Unsupported nccl-tests register mode: {register_mode}")
    if register_value:
        args.extend(["-R", str(register_value)])

    if config.cudagraph_launches < 0:
        raise ValueError("cudagraph launches must be >= 0")
    if config.cudagraph_launches:
        args.extend(["-G", str(config.cudagraph_launches)])

    if config.cta_policy is not None:
        if config.cta_policy not in (0, 1, 2):
            raise ValueError(f"Unsupported CTA policy: {config.cta_policy}")
        args.extend(["-x", str(config.cta_policy)])

    device_impl = config.effective_device_impl()
    if device_impl < 0:
        raise ValueError("device implementation must be >= 0")
    if device_impl:
        if register_mode != "symmetric":
            raise ValueError("device implementation requires symmetric registration")
        args.extend(["-D", str(device_impl)])
        if config.device_cta_count <= 0:
            raise ValueError("device CTA count must be > 0")
        args.extend(["-V", str(config.device_cta_count)])

    blocking_mode = config.effective_blocking_mode()
    if blocking_mode is not None:
        if blocking_mode not in (1, 2):
            raise ValueError(f"Unsupported blocking mode: {blocking_mode}")
        args.extend(["-z", str(blocking_mode)])

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
