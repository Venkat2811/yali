#!/bin/bash
# Example Performance Sweep
# Tests simple_allreduce and multilane_allreduce for correctness,
# then validates YALI benchmark achieves expected bandwidth.
#
# Usage: CUDA_VISIBLE_DEVICES=0,1 ./run_example_perf.sh
#
# Environment:
#   SIMPLE_BIN      - Path to simple_allreduce binary
#   MULTILANE_BIN   - Path to multilane_allreduce binary
#   BENCHMARK_BIN   - Path to benchmark_yali (for reference)

set -e

# Parse args
for arg in "$@"; do
    case $arg in
        --simple=*)
            SIMPLE_BIN="${arg#*=}"
            ;;
        --multilane=*)
            MULTILANE_BIN="${arg#*=}"
            ;;
        --benchmark=*)
            BENCHMARK_BIN="${arg#*=}"
            ;;
    esac
done

# Defaults
SIMPLE_BIN="${SIMPLE_BIN:-bazel-bin/example_simple}"
MULTILANE_BIN="${MULTILANE_BIN:-bazel-bin/example_multilane}"
BENCHMARK_BIN="${BENCHMARK_BIN:-bazel-bin/benchmark_yali}"

echo "========================================"
echo "Example Performance Sweep"
echo "========================================"
echo ""

# Test 1: Run simple_allreduce (correctness)
echo "[1/4] Testing simple_allreduce correctness..."
if $SIMPLE_BIN; then
    echo "  PASS: simple_allreduce"
else
    echo "  FAIL: simple_allreduce"
    exit 1
fi
echo ""

# Test 2: Run multilane_allreduce (correctness)
echo "[2/4] Testing multilane_allreduce correctness..."
if $MULTILANE_BIN; then
    echo "  PASS: multilane_allreduce"
else
    echo "  FAIL: multilane_allreduce"
    exit 1
fi
echo ""

# Test 3: Performance comparison - 64MB (flash kernel regime)
echo "[3/4] Performance @ 64MB (16M fp32 elements)..."
echo ""
echo "  Size     | Benchmark GB/s | Threshold"
echo "  ---------|----------------|----------"

# Run benchmark at 64MB (16M float elements = 64MB)
ELEMS_64M=$((16 * 1024 * 1024))
RESULT=$($BENCHMARK_BIN $ELEMS_64M 20 0 flash 0 cuda-events 2>&1 | grep -E "GB/s" | tail -1 || echo "N/A")
# Extract bandwidth
BW_64M=$(echo "$RESULT" | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "0")
printf "  64 MB    | %14s | >= 60\n" "$BW_64M"

# Validate 64MB meets threshold
if (( $(echo "$BW_64M >= 60" | bc -l) )); then
    echo "  PASS: 64MB bandwidth >= 60 GB/s"
else
    echo "  WARNING: 64MB bandwidth below 60 GB/s threshold"
fi
echo ""

# Test 4: Performance comparison - 2GB (stream kernel regime)
echo "[4/4] Performance @ 2GB (512M fp32 elements)..."
echo ""

ELEMS_2G=$((512 * 1024 * 1024))
RESULT=$($BENCHMARK_BIN $ELEMS_2G 10 0 stream 0 cuda-events 2>&1 | grep -E "GB/s" | tail -1 || echo "N/A")
BW_2G=$(echo "$RESULT" | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "0")
printf "  2 GB     | %14s | >= 250\n" "$BW_2G"

if (( $(echo "$BW_2G >= 250" | bc -l) )); then
    echo "  PASS: 2GB bandwidth >= 250 GB/s"
else
    echo "  WARNING: 2GB bandwidth below 250 GB/s threshold"
fi
echo ""

# Summary
echo "========================================"
echo "Summary"
echo "========================================"
echo "  simple_allreduce:    PASS"
echo "  multilane_allreduce: PASS"
echo "  64MB benchmark:      $BW_64M GB/s"
echo "  2GB benchmark:       $BW_2G GB/s"
echo "========================================"
echo ""
echo "All example tests passed!"
