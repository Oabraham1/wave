#!/bin/bash
# Copyright 2026 Ojima Abraham
# SPDX-License-Identifier: Apache-2.0
# Backend codegen verification: checks output contains expected constructs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

ASM="$ROOT_DIR/wave-asm/target/release/wave-asm"
METAL="$ROOT_DIR/wave-metal/target/release/wave-metal"
PTX="$ROOT_DIR/wave-ptx/target/release/wave-ptx"
HIP="$ROOT_DIR/wave-hip/target/release/wave-hip"
SYCL="$ROOT_DIR/wave-sycl/target/release/wave-sycl"

PASS=0
FAIL=0
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

fail_msg() {
    echo "[FAIL] $1"
    if [ -n "${2:-}" ]; then echo "  Detail: $2"; fi
    FAIL=$((FAIL + 1))
}

pass_msg() {
    echo "[PASS] $1"
    PASS=$((PASS + 1))
}

check_contains() {
    grep -q "$2" "$1"
}

for crate in wave-decode wave-asm wave-metal wave-ptx wave-hip wave-sycl; do
    (cd "$ROOT_DIR/$crate" && cargo build --release 2>&1) | tail -1
done

"$ASM" "$ROOT_DIR/wave-metal/examples/vector_add.wave" -o "$TMPDIR/va.wbin"

"$METAL" "$TMPDIR/va.wbin" -o "$TMPDIR/va.metal"
metal_ok=true
for pattern in "kernel void" "thread_position_in_threadgroup" "device.*uint8_t" "rf(" ; do
    if ! check_contains "$TMPDIR/va.metal" "$pattern"; then
        fail_msg "Metal codegen" "missing pattern: $pattern"
        metal_ok=false
        break
    fi
done
$metal_ok && pass_msg "Metal codegen (vector_add)"

"$PTX" "$TMPDIR/va.wbin" -o "$TMPDIR/va.ptx"
ptx_ok=true
for pattern in ".entry" "%tid" ".reg" "add.f32" ; do
    if ! check_contains "$TMPDIR/va.ptx" "$pattern"; then
        fail_msg "PTX codegen" "missing pattern: $pattern"
        ptx_ok=false
        break
    fi
done
$ptx_ok && pass_msg "PTX codegen (vector_add)"

"$HIP" "$TMPDIR/va.wbin" -o "$TMPDIR/va.hip"
hip_ok=true
for pattern in "__global__" "threadIdx" "hip/hip_runtime.h" ; do
    if ! check_contains "$TMPDIR/va.hip" "$pattern"; then
        fail_msg "HIP codegen" "missing pattern: $pattern"
        hip_ok=false
        break
    fi
done
$hip_ok && pass_msg "HIP codegen (vector_add)"

"$SYCL" "$TMPDIR/va.wbin" -o "$TMPDIR/va.sycl"
sycl_ok=true
for pattern in "sycl/sycl.hpp" "parallel_for" "nd_range" ; do
    if ! check_contains "$TMPDIR/va.sycl" "$pattern"; then
        fail_msg "SYCL codegen" "missing pattern: $pattern"
        sycl_ok=false
        break
    fi
done
$sycl_ok && pass_msg "SYCL codegen (vector_add)"

echo ""
echo "Results: $PASS passed, $FAIL failed"
exit "$FAIL"
