#!/bin/bash
# Validate nvbandwidth runs correctly and measures NVLink bandwidth
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find the binary - it should be in runfiles
RUNFILES_DIR="${BASH_SOURCE[0]}.runfiles"
if [[ -d "$RUNFILES_DIR" ]]; then
    NVBANDWIDTH="$RUNFILES_DIR/_main/nvbandwidth"
else
    # Fallback to bazel-bin location
    NVBANDWIDTH="${BUILD_WORKSPACE_DIRECTORY:-$PWD}/bazel-bin/nvbandwidth"
fi

if [[ ! -x "$NVBANDWIDTH" ]]; then
    echo "ERROR: nvbandwidth not found at $NVBANDWIDTH"
    exit 1
fi

MIN_NVLINK_BW_GBPS="${MIN_NVLINK_BW_GBPS:-100}"

echo "=== Validating nvbandwidth ==="
echo "Binary: $NVBANDWIDTH"
echo "Minimum expected NVLink bandwidth: ${MIN_NVLINK_BW_GBPS} GB/s"
echo ""

# Check version
echo "nvbandwidth version:"
"$NVBANDWIDTH" --version 2>&1 || true
echo ""

# Run device-to-device memcpy test
echo "Running device_to_device_memcpy_read_ce test..."
OUTPUT=$("$NVBANDWIDTH" -t device_to_device_memcpy_read_ce 2>&1)
echo "$OUTPUT"
echo ""

# Extract SUM bandwidth (total of all GPU pairs)
SUM_BW=$(echo "$OUTPUT" | grep "^SUM" | grep -o '[0-9.]*' | head -1)
if [[ -z "$SUM_BW" ]]; then
    echo "FAILED: Could not extract bandwidth from nvbandwidth output"
    exit 1
fi

# For 2 GPUs, we have 2 measurements (0->1 and 1->0), so divide by 2 for single direction
# Actually, SUM is total of all pairs, and each pair is unidirectional
# So for 2 GPUs: SUM = bw(0->1) + bw(1->0) which are both ~140 GB/s
# We check per-direction bandwidth by dividing SUM by 2
PER_DIR_BW=$(echo "scale=2; $SUM_BW / 2" | bc)

echo "Measured NVLink bandwidth:"
echo "  Total (SUM): ${SUM_BW} GB/s"
echo "  Per direction: ${PER_DIR_BW} GB/s"

# Compare with minimum expected
if (( $(echo "$PER_DIR_BW < $MIN_NVLINK_BW_GBPS" | bc -l) )); then
    echo "FAILED: NVLink bandwidth ($PER_DIR_BW GB/s) is below minimum expected ($MIN_NVLINK_BW_GBPS GB/s)"
    exit 1
fi

echo ""
echo "PASSED: nvbandwidth validation successful"
echo "  Per-direction NVLink bandwidth: ${PER_DIR_BW} GB/s (minimum: ${MIN_NVLINK_BW_GBPS} GB/s)"
