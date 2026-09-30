#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Generate, compile, validate, and measure local Metal GEMM configurations.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SCRIPT_DIR/run_autotune.py" "$@"
