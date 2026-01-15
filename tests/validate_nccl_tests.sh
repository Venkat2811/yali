#!/bin/bash
# Validate nccl-tests runs correctly and achieves expected performance
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find the binaries - they should be in runfiles
RUNFILES_DIR="${BASH_SOURCE[0]}.runfiles"
if [[ -d "$RUNFILES_DIR" ]]; then
    ALL_REDUCE_PERF="$RUNFILES_DIR/_main/all_reduce_perf"
    LIBNCCL_SO="$RUNFILES_DIR/_main/libnccl.so.2"
else
    # Fallback to bazel-bin location
    ALL_REDUCE_PERF="${BUILD_WORKSPACE_DIRECTORY:-$PWD}/bazel-bin/all_reduce_perf"
    LIBNCCL_SO="${BUILD_WORKSPACE_DIRECTORY:-$PWD}/bazel-bin/libnccl.so.2"
fi

if [[ ! -x "$ALL_REDUCE_PERF" ]]; then
    echo "ERROR: all_reduce_perf not found at $ALL_REDUCE_PERF"
    exit 1
fi

if [[ ! -f "$LIBNCCL_SO" ]]; then
    echo "ERROR: libnccl.so.2 not found at $LIBNCCL_SO"
    exit 1
fi

MIN_BUS_BW_GBPS="${MIN_BUS_BW_GBPS:-80}"

echo "=== Validating NCCL Tests Performance ==="
echo "Binary: $ALL_REDUCE_PERF"
echo "Library: $LIBNCCL_SO"
echo "Minimum expected bus bandwidth: ${MIN_BUS_BW_GBPS} GB/s"
echo ""

# Set LD_LIBRARY_PATH to use our libnccl.so.2
export LD_LIBRARY_PATH="$(dirname "$LIBNCCL_SO"):$LD_LIBRARY_PATH"

# Run performance test at 128M
echo "Running all_reduce_perf at 128M..."
OUTPUT=$("$ALL_REDUCE_PERF" -g 2 -b 128M -e 128M -w 1 -n 5 2>&1)
echo "$OUTPUT"
echo ""

# Check for validation errors
if echo "$OUTPUT" | grep -q "FAILED"; then
    echo "FAILED: nccl-tests validation failed"
    exit 1
fi

# Check for out of bounds errors
OOB_COUNT=$(echo "$OUTPUT" | grep "Out of bounds" | grep -o '[0-9]*' | head -1)
if [[ "$OOB_COUNT" != "0" ]]; then
    echo "FAILED: Out of bounds values detected: $OOB_COUNT"
    exit 1
fi

# Extract average bus bandwidth
AVG_BW=$(echo "$OUTPUT" | grep "Avg bus bandwidth" | grep -o '[0-9.]*' | head -1)
if [[ -z "$AVG_BW" ]]; then
    echo "FAILED: Could not extract average bus bandwidth"
    exit 1
fi

echo "Measured average bus bandwidth: ${AVG_BW} GB/s"

# Compare with minimum expected (using bc for floating point)
if (( $(echo "$AVG_BW < $MIN_BUS_BW_GBPS" | bc -l) )); then
    echo "FAILED: Bus bandwidth ($AVG_BW GB/s) is below minimum expected ($MIN_BUS_BW_GBPS GB/s)"
    exit 1
fi

echo ""
echo "PASSED: NCCL tests validation successful"
echo "  Bus bandwidth: ${AVG_BW} GB/s (minimum: ${MIN_BUS_BW_GBPS} GB/s)"
