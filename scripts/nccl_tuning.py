#!/usr/bin/env python3
"""
Shared NCCL tuning helpers for YALI benchmark scripts.

This keeps NCCL tuning logic centralized so the quick benchmark and the full
sweep use the same candidate matrix, runtime environment, and cache behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class NcclCandidate:
    name: str
    env_items: Tuple[Tuple[str, str], ...]

    def env(self) -> Dict[str, str]:
        return dict(self.env_items)


DEFAULT_CANDIDATE = NcclCandidate("default", ())


def _candidate(name: str, **env: str) -> NcclCandidate:
    return NcclCandidate(name, tuple(sorted(env.items())))


SMALL_CANDIDATES = (
    DEFAULT_CANDIDATE,
    _candidate("ring-ll", NCCL_ALGO="RING", NCCL_PROTO="LL"),
    _candidate("ring-ll128", NCCL_ALGO="RING", NCCL_PROTO="LL128"),
    _candidate("tree-ll", NCCL_ALGO="TREE", NCCL_PROTO="LL"),
    _candidate("tree-ll128", NCCL_ALGO="TREE", NCCL_PROTO="LL128"),
)

MEDIUM_CANDIDATES = (
    DEFAULT_CANDIDATE,
    _candidate("ring-ll128", NCCL_ALGO="RING", NCCL_PROTO="LL128"),
    _candidate("ring-simple", NCCL_ALGO="RING", NCCL_PROTO="SIMPLE"),
    _candidate("tree-ll128", NCCL_ALGO="TREE", NCCL_PROTO="LL128"),
    _candidate("tree-simple", NCCL_ALGO="TREE", NCCL_PROTO="SIMPLE"),
)

LARGE_CANDIDATES = (
    DEFAULT_CANDIDATE,
    _candidate("ring-simple", NCCL_ALGO="RING", NCCL_PROTO="SIMPLE"),
    _candidate("tree-simple", NCCL_ALGO="TREE", NCCL_PROTO="SIMPLE"),
    _candidate(
        "ring-simple-ch2",
        NCCL_ALGO="RING",
        NCCL_PROTO="SIMPLE",
        NCCL_MIN_NCHANNELS="2",
        NCCL_MAX_NCHANNELS="2",
    ),
    _candidate(
        "ring-simple-ch4",
        NCCL_ALGO="RING",
        NCCL_PROTO="SIMPLE",
        NCCL_MIN_NCHANNELS="4",
        NCCL_MAX_NCHANNELS="4",
    ),
)

SMALL_CANDIDATES_EXTENDED = SMALL_CANDIDATES + (
    _candidate("ring-default", NCCL_ALGO="RING"),
    _candidate("tree-default", NCCL_ALGO="TREE"),
)

MEDIUM_CANDIDATES_EXTENDED = MEDIUM_CANDIDATES + (
    _candidate("ring-default", NCCL_ALGO="RING"),
    _candidate("tree-default", NCCL_ALGO="TREE"),
    _candidate(
        "ring-simple-ch8",
        NCCL_ALGO="RING",
        NCCL_PROTO="SIMPLE",
        NCCL_MIN_NCHANNELS="8",
        NCCL_MAX_NCHANNELS="8",
    ),
)

LARGE_CANDIDATES_EXTENDED = LARGE_CANDIDATES + (
    _candidate("ring-default", NCCL_ALGO="RING"),
    _candidate("tree-default", NCCL_ALGO="TREE"),
    _candidate(
        "tree-simple-ch2",
        NCCL_ALGO="TREE",
        NCCL_PROTO="SIMPLE",
        NCCL_MIN_NCHANNELS="2",
        NCCL_MAX_NCHANNELS="2",
    ),
    _candidate(
        "tree-simple-ch4",
        NCCL_ALGO="TREE",
        NCCL_PROTO="SIMPLE",
        NCCL_MIN_NCHANNELS="4",
        NCCL_MAX_NCHANNELS="4",
    ),
)


def candidate_env_string(candidate: NcclCandidate) -> str:
    if not candidate.env_items:
        return "(default)"
    return " ".join(f"{key}={value}" for key, value in candidate.env_items)


class NcclTuner:
    def __init__(self, project_dir: Path, mode: str = "default", level: str = "basic"):
        self.project_dir = Path(project_dir)
        self.mode = mode
        self.level = level
        self.nccl_lib_dir = self.project_dir / "third_party" / "nccl" / "build" / "lib"
        self._selection_cache: Dict[Tuple[object, ...], NcclCandidate] = {}
        self._selection_details: Dict[Tuple[object, ...], Dict[str, object]] = {}

    def enabled(self) -> bool:
        return self.mode == "tuned"

    def prepare_runtime_env(self, env: Dict[str, str]) -> Dict[str, str]:
        merged = dict(env)
        merged["LD_LIBRARY_PATH"] = f"{self.nccl_lib_dir}:{merged.get('LD_LIBRARY_PATH', '')}"
        return merged

    def get_selection(self, key: Tuple[object, ...]) -> Optional[NcclCandidate]:
        return self._selection_cache.get(key)

    def selection_records(self) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        for key, candidate in sorted(self._selection_cache.items(), key=lambda item: item[0]):
            detail = self._selection_details.get(key, {})
            records.append(
                {
                    "benchmark_key": [str(part) for part in key],
                    "selected": candidate.name,
                    "selected_env": candidate_env_string(candidate),
                    "size_bytes": detail.get("size_bytes", 0),
                    "candidates": detail.get("candidates", []),
                }
            )
        return records

    def resolve(
        self,
        key: Tuple[object, ...],
        size_bytes: int,
        base_env: Dict[str, str],
        runner: Callable[[Dict[str, str]], Optional[float]],
    ) -> Tuple[Dict[str, str], str]:
        runtime_env = self.prepare_runtime_env(base_env)
        if not self.enabled():
            return runtime_env, DEFAULT_CANDIDATE.name

        if key not in self._selection_cache:
            selected = self._select_candidate(key, size_bytes, runtime_env, runner)
            self._selection_cache[key] = selected

        candidate = self._selection_cache[key]
        tuned_env = dict(runtime_env)
        tuned_env.update(candidate.env())
        return tuned_env, candidate.name

    def _select_candidate(
        self,
        key: Tuple[object, ...],
        size_bytes: int,
        runtime_env: Dict[str, str],
        runner: Callable[[Dict[str, str]], Optional[float]],
    ) -> NcclCandidate:
        best = DEFAULT_CANDIDATE
        best_score = -1.0
        attempts: List[Dict[str, object]] = []

        for candidate in self._candidates_for_size(size_bytes):
            tuned_env = dict(runtime_env)
            tuned_env.update(candidate.env())
            score = runner(tuned_env)
            attempts.append(
                {
                    "name": candidate.name,
                    "env": candidate_env_string(candidate),
                    "gbps": score,
                }
            )
            if score is not None and score > best_score:
                best = candidate
                best_score = score

        self._selection_details[key] = {
            "size_bytes": size_bytes,
            "candidates": attempts,
        }
        return best

    def _candidates_for_size(self, size_bytes: int) -> Tuple[NcclCandidate, ...]:
        if self.level == "extended":
            if size_bytes <= 4 * 1024 * 1024:
                return SMALL_CANDIDATES_EXTENDED
            if size_bytes <= 128 * 1024 * 1024:
                return MEDIUM_CANDIDATES_EXTENDED
            return LARGE_CANDIDATES_EXTENDED

        if size_bytes <= 4 * 1024 * 1024:
            return SMALL_CANDIDATES
        if size_bytes <= 128 * 1024 * 1024:
            return MEDIUM_CANDIDATES
        return LARGE_CANDIDATES
