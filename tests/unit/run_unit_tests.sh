#!/usr/bin/env bash
# Simple unit tests for Yali harness
# Tests different configurations via env vars and checks for PASSED output
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR="${BUILD_WORKSPACE_DIRECTORY:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

# Find binary
if [[ -n "${YALI_BIN:-}" ]]; then
  BIN="$YALI_BIN"
elif [[ -f "$ROOT_DIR/bin/yali_harness" ]]; then
  BIN="$ROOT_DIR/bin/yali_harness"
else
  BAZEL_BIN=$(bazel info bazel-bin 2>/dev/null || echo "")
  if [[ -n "$BAZEL_BIN" && -f "$BAZEL_BIN/yali_harness" ]]; then
    BIN="$BAZEL_BIN/yali_harness"
  else
    echo "ERROR: Cannot find yali harness binary. Run 'make build-yali' first." >&2
    exit 1
  fi
fi

echo "Using binary: $BIN"
echo "========================================"

PASSED=0
FAILED=0

run_test() {
  local name="$1"
  shift
  echo -n "[ RUN  ] $name ... "

  local output
  if output=$("$@" 2>&1); then
    if echo "$output" | grep -q "PASSED"; then
      echo "OK"
      ((PASSED++))
      return 0
    else
      echo "FAIL (no PASSED in output)"
      echo "$output" | tail -5
      ((FAILED++))
      return 1
    fi
  else
    echo "FAIL (exit code $?)"
    echo "$output" | tail -5
    ((FAILED++))
    return 1
  fi
}

# GPU check
if ! nvidia-smi >/dev/null 2>&1; then
  echo "SKIP: No GPUs available"
  exit 0
fi

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
if [[ $GPU_COUNT -lt 2 ]]; then
  echo "SKIP: Need 2 GPUs, found $GPU_COUNT"
  exit 0
fi

export CUDA_VISIBLE_DEVICES=0,1

echo ""
echo "=== Dtype Tests ==="
run_test "FP32 small" env YALI_ELEMS=1024 YALI_DTYPE=fp32 YALI_DEV_WARMUP=0 YALI_DEV_ITERS=1 "$BIN" || true
run_test "FP16 small" env YALI_ELEMS=1024 YALI_DTYPE=fp16 YALI_DEV_WARMUP=0 YALI_DEV_ITERS=1 "$BIN" || true
run_test "BF16 small" env YALI_ELEMS=1024 YALI_DTYPE=bf16 YALI_DEV_WARMUP=0 YALI_DEV_ITERS=1 "$BIN" || true

echo ""
echo "=== Size Tests (FP32) ==="
run_test "1K elems" env YALI_ELEMS=1024 YALI_DTYPE=fp32 YALI_DEV_WARMUP=0 YALI_DEV_ITERS=1 "$BIN" || true
run_test "64K elems" env YALI_ELEMS=65536 YALI_DTYPE=fp32 YALI_DEV_WARMUP=0 YALI_DEV_ITERS=1 "$BIN" || true
run_test "256K elems" env YALI_ELEMS=262144 YALI_DTYPE=fp32 YALI_DEV_WARMUP=0 YALI_DEV_ITERS=1 "$BIN" || true
run_test "1M elems" env YALI_ELEMS=1048576 YALI_DTYPE=fp32 YALI_DEV_WARMUP=0 YALI_DEV_ITERS=1 "$BIN" || true

echo ""
echo "=== Kernel Mode Tests ==="
run_test "auto mode" env YALI_ELEMS=262144 YALI_KERNEL_MODE=auto YALI_DEV_WARMUP=0 YALI_DEV_ITERS=1 "$BIN" || true
run_test "flash mode" env YALI_ELEMS=262144 YALI_KERNEL_MODE=flash YALI_DEV_WARMUP=0 YALI_DEV_ITERS=1 "$BIN" || true
run_test "bandwidth mode" env YALI_ELEMS=262144 YALI_KERNEL_MODE=bandwidth YALI_DEV_WARMUP=0 YALI_DEV_ITERS=1 "$BIN" || true

echo ""
echo "=== Lane Count Tests ==="
run_test "4 lanes" env YALI_ELEMS=262144 YALI_DEV_LANES=4 YALI_DEV_WARMUP=0 YALI_DEV_ITERS=1 "$BIN" || true
run_test "16 lanes" env YALI_ELEMS=262144 YALI_DEV_LANES=16 YALI_DEV_WARMUP=0 YALI_DEV_ITERS=1 "$BIN" || true
run_test "64 lanes" env YALI_ELEMS=262144 YALI_DEV_LANES=64 YALI_DEV_WARMUP=0 YALI_DEV_ITERS=1 "$BIN" || true

echo ""
echo "========================================"
echo "Results: $PASSED passed, $FAILED failed"

if [[ $FAILED -gt 0 ]]; then
  exit 1
fi
echo "All unit tests PASSED"
