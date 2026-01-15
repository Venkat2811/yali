#!/usr/bin/env bash
# Build script for Yali examples
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR="${BUILD_WORKSPACE_DIRECTORY:-$(cd "$SCRIPT_DIR/.." && pwd)}"
OUT_DIR="$ROOT_DIR/bin/examples"
mkdir -p "$OUT_DIR"

NVCC=${NVCC:-nvcc}
CUDA_ARCH="${CUDA_ARCH:-80}"

NVCC_FLAGS=(
  -std=c++17
  -rdc=true
  --expt-extended-lambda
  -lineinfo
  -gencode "arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}"
  -I"$ROOT_DIR"
  -I"$ROOT_DIR/include"
  -I"$ROOT_DIR/src/kernels"
  -I"$ROOT_DIR/src"
  -I"$ROOT_DIR/third_party/nccl/src"
  -I"$ROOT_DIR/third_party/nccl/src/include"
  -I"$ROOT_DIR/third_party/nccl/src/include/plugin"
  -I"$ROOT_DIR/third_party/nccl/src/device"
)

build_example() {
  local name="$1"
  local src="$2"
  echo "Building $name..."
  "$NVCC" "${NVCC_FLAGS[@]}" "$src" "$ROOT_DIR/src/kernels/bandwidth.cu" \
    -lcudadevrt -o "$OUT_DIR/$name"
  echo "Built: $OUT_DIR/$name"
}

EXAMPLE_NAME="${1:-all}"

case "$EXAMPLE_NAME" in
  basic_all_reduce)
    build_example "basic_all_reduce" "$SCRIPT_DIR/basic_all_reduce.cu"
    ;;
  all)
    build_example "basic_all_reduce" "$SCRIPT_DIR/basic_all_reduce.cu"
    ;;
  *)
    echo "Unknown example: $EXAMPLE_NAME"
    echo "Usage: $0 [basic_all_reduce|all]"
    exit 1
    ;;
esac

echo ""
echo "To run: CUDA_VISIBLE_DEVICES=0,1 $OUT_DIR/<example_name>"
