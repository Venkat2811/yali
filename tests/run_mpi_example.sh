#!/usr/bin/env bash
#
# Run an MPI example binary with mpirun
#
# Usage: run_mpi_example.sh <binary_path>
#
set -euo pipefail

BINARY="${1:?ERROR: Binary path required}"
NPROCS="${NPROCS:-2}"

# Ensure binary exists
if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: Binary not found or not executable: $BINARY"
    exit 1
fi

# Run with mpirun
echo "Running: mpirun -np $NPROCS --allow-run-as-root $BINARY"
mpirun -np "$NPROCS" --allow-run-as-root --bind-to none \
    -x CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
    "$BINARY"

exit $?
