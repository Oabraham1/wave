#!/bin/bash
# Copyright (c) 2026 Ojima Abraham. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "WAVE Spec Verification Suite"
echo "============================"
echo

echo "Building toolchain..."
cd "$REPO_ROOT"

cargo build --release -p wave-asm -p wave-emu -p wave-decode 2>/dev/null || {
    echo "ERROR: Failed to build toolchain"
    exit 1
}

echo "Toolchain built successfully."
echo

echo "Building verification suite..."
cd "$SCRIPT_DIR"

cargo build --release 2>/dev/null || {
    echo "ERROR: Failed to build verification suite"
    exit 1
}

echo "Running specification verification tests..."
echo

./target/release/run-spec-tests
EXIT_CODE=$?

echo
if [ $EXIT_CODE -eq 0 ]; then
    echo "All tests PASSED"
else
    echo "Some tests FAILED (see report above)"
fi

exit $EXIT_CODE
