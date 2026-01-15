#!/bin/bash
# Validate NCCL header/library version matching
# This test ensures nccl-tests was built against our NCCL submodule, not system NCCL
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments for binary location
ALL_REDUCE_PERF=""
LIBNCCL_SO=""
for arg in "$@"; do
    case $arg in
        --binary=*)
            # Not used for this test
            ;;
    esac
done

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

EXPECTED_NCCL_VERSION="${EXPECTED_NCCL_VERSION:-22809}"

echo "=== Validating NCCL Version Matching ==="
echo "Expected NCCL version: $EXPECTED_NCCL_VERSION"
echo "Binary: $ALL_REDUCE_PERF"
echo "Library: $LIBNCCL_SO"
echo ""

# Set LD_LIBRARY_PATH to use our libnccl.so.2
export LD_LIBRARY_PATH="$(dirname "$LIBNCCL_SO"):$LD_LIBRARY_PATH"

# Verify which libnccl is being loaded
echo "Checking library linking..."
LDD_OUTPUT=$(ldd "$ALL_REDUCE_PERF" | grep libnccl || true)
echo "Linked NCCL: $LDD_OUTPUT"

if [[ "$LDD_OUTPUT" != *"$LIBNCCL_SO"* ]] && [[ "$LDD_OUTPUT" != *"libnccl.so.2"* ]]; then
    echo "WARNING: NCCL library path may not be correct"
fi
echo ""

# Run all_reduce_perf to get version info
echo "Running nccl-tests to check version..."
OUTPUT=$("$ALL_REDUCE_PERF" -g 2 -b 1M -e 1M -w 0 -n 1 2>&1 | head -5)
echo "$OUTPUT"
echo ""

# Extract header and library versions
HEADER_VERSION=$(echo "$OUTPUT" | grep -o 'nccl-headers=[0-9]*' | cut -d= -f2)
LIBRARY_VERSION=$(echo "$OUTPUT" | grep -o 'nccl-library=[0-9]*' | cut -d= -f2)

echo "Detected versions:"
echo "  Header version:  $HEADER_VERSION"
echo "  Library version: $LIBRARY_VERSION"
echo "  Expected:        $EXPECTED_NCCL_VERSION"
echo ""

# Validate versions match
if [[ "$HEADER_VERSION" != "$LIBRARY_VERSION" ]]; then
    echo "FAILED: Header version ($HEADER_VERSION) does not match library version ($LIBRARY_VERSION)"
    echo ""
    echo "This indicates nccl-tests was built against system NCCL headers instead of our submodule."
    echo "Rebuild with: bazel clean && bazel build //:nccl_tests_bin"
    exit 1
fi

if [[ "$HEADER_VERSION" != "$EXPECTED_NCCL_VERSION" ]]; then
    echo "FAILED: NCCL version ($HEADER_VERSION) does not match expected ($EXPECTED_NCCL_VERSION)"
    echo ""
    echo "This may indicate the NCCL submodule is at a different version."
    echo "Check: cd nccl && git describe --tags"
    exit 1
fi

echo "PASSED: NCCL version validation successful"
echo "  nccl-headers=$HEADER_VERSION nccl-library=$LIBRARY_VERSION"
