#!/usr/bin/env bash
# Simple unit test runner
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <test_binary>" >&2
  exit 1
fi

TEST_BIN="$1"

if [[ ! -x "$TEST_BIN" ]]; then
  echo "ERROR: Test binary not found or not executable: $TEST_BIN" >&2
  exit 4
fi

# Run the test
"$TEST_BIN"
exit_code=$?

if [[ $exit_code -eq 0 ]]; then
  echo "Unit test PASSED"
else
  echo "Unit test FAILED (exit code: $exit_code)"
fi

exit $exit_code
