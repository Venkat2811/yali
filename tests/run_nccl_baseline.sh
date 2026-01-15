#!/bin/bash
# Run NCCL baseline tests with different algorithms
# Usage: ./run_nccl_baseline.sh --algo <RING|TREE|DEVICE_API> --dtype <float|half|bfloat16>
#
# Environment variables:
#   NCCL_TEST_BIN      - Path to all_reduce_perf binary (required)
#   CUDA_VISIBLE_DEVICES - GPUs to use (default: 0,1)
#   NCCL_WARMUP        - Warmup iterations (default: 1)
#   NCCL_ITERS         - Measurement iterations (default: 5)
#   BASELINE_SIZES     - Space-separated sizes (default: full sweep)
#   TEST_UNDECLARED_OUTPUTS_DIR - Output directory

set -euo pipefail

# Defaults
ALGO="${NCCL_ALGO:-RING}"
DTYPE="${NCCL_DTYPE:-float}"
WARMUP="${NCCL_WARMUP:-1}"
ITERS="${NCCL_ITERS:-5}"
SIZES="${BASELINE_SIZES:-2K 4K 8K 16K 32K 64K 128K 256K 512K 1M 2M 4M 8M 16M 32M 64M 128M 256M 512M 1G 2G}"
OUT_DIR="${TEST_UNDECLARED_OUTPUTS_DIR:-$(pwd)}"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --algo) ALGO="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Validate
if [[ -z "${NCCL_TEST_BIN:-}" ]]; then
    echo "ERROR: NCCL_TEST_BIN not set" >&2
    exit 1
fi

if [[ ! -x "$NCCL_TEST_BIN" ]]; then
    echo "ERROR: NCCL_TEST_BIN not executable: $NCCL_TEST_BIN" >&2
    exit 1
fi

# Set LD_LIBRARY_PATH for libnccl.so.2
NCCL_LIB_DIR=$(dirname "$NCCL_TEST_BIN")
export LD_LIBRARY_PATH="${NCCL_LIB_DIR}:${LD_LIBRARY_PATH:-}"

# Build command based on algorithm
CMD_ARGS="-g 2 -b 2K -e 2G -f 2 -w $WARMUP -n $ITERS -d $DTYPE"

case "$ALGO" in
    RING)
        export NCCL_ALGO=RING
        OUTPUT_FILE="$OUT_DIR/nccl_ring_${DTYPE}.txt"
        ;;
    TREE)
        export NCCL_ALGO=TREE
        OUTPUT_FILE="$OUT_DIR/nccl_tree_${DTYPE}.txt"
        ;;
    DEVICE_API|DEVAPI)
        # Device API requires symmetric memory and blocking for proper timing
        CMD_ARGS="$CMD_ARGS -D 2 -R 2 -z 1"
        OUTPUT_FILE="$OUT_DIR/nccl_devapi_${DTYPE}.txt"
        ;;
    *)
        echo "ERROR: Unknown algorithm: $ALGO (use RING, TREE, or DEVICE_API)" >&2
        exit 1
        ;;
esac

echo "=============================================="
echo "NCCL Baseline: $ALGO ($DTYPE)"
echo "=============================================="
echo "Binary: $NCCL_TEST_BIN"
echo "Output: $OUTPUT_FILE"
echo "Command: $NCCL_TEST_BIN $CMD_ARGS"
echo ""

mkdir -p "$OUT_DIR"

# Run benchmark
"$NCCL_TEST_BIN" $CMD_ARGS 2>&1 | tee "$OUTPUT_FILE"

echo ""
echo "Results saved to: $OUTPUT_FILE"
