#!/bin/bash
# Copyright 2026 Ojima Abraham
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec cargo run --locked --release --manifest-path "$SCRIPT_DIR/Cargo.toml" -- "$@"
