#!/bin/bash
# Build NCCL and nccl-tests with proper header isolation
# Target: sm_80 (A100)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NCCL_DIR="$ROOT_DIR/nccl"
NCCL_TESTS_DIR="$ROOT_DIR/nccl-tests"
OUTPUT_DIR="$ROOT_DIR/bin"

# SM architecture (A100 = sm_80)
SM_ARCH="${SM_ARCH:-80}"
NVCC_GENCODE="-gencode=arch=compute_${SM_ARCH},code=sm_${SM_ARCH}"

echo "=== Building NCCL v2.28.9 (sm_${SM_ARCH}) ==="
cd "$NCCL_DIR"
make -j$(nproc) src.build NVCC_GENCODE="$NVCC_GENCODE"

NCCL_BUILD="$NCCL_DIR/build"
echo "NCCL built at: $NCCL_BUILD"
echo "NCCL header version: $(grep NCCL_VERSION_CODE $NCCL_BUILD/include/nccl.h | head -1)"

echo ""
echo "=== Building nccl-tests v2.17.6 (sm_${SM_ARCH}) ==="
cd "$NCCL_TESTS_DIR"
rm -rf build

# Build with explicit include path using -isystem to override system headers
make -j$(nproc) MPI=0 \
    CUDA_HOME="$CUDA_HOME" \
    NCCL_HOME="$NCCL_BUILD" \
    NVCC_GENCODE="$NVCC_GENCODE" \
    NVCUFLAGS="-ccbin g++ $NVCC_GENCODE -std=c++17 -O3 -g -isystem $NCCL_BUILD/include"

echo ""
echo "=== Copying binaries to $OUTPUT_DIR ==="
mkdir -p "$OUTPUT_DIR"
cp build/all_reduce_perf "$OUTPUT_DIR/"
cp "$NCCL_BUILD/lib/libnccl.so.2" "$OUTPUT_DIR/" 2>/dev/null || true

echo ""
echo "=== Build complete ==="
echo "Binary: $OUTPUT_DIR/all_reduce_perf"
echo "Library: $OUTPUT_DIR/libnccl.so.2"
echo ""
echo "To run: LD_LIBRARY_PATH=$NCCL_BUILD/lib:\$LD_LIBRARY_PATH $OUTPUT_DIR/all_reduce_perf -g 2 -b 128M -e 128M"
