#!/bin/bash
# Copyright 2026 Ojima Abraham
# SPDX-License-Identifier: Apache-2.0
# Emulator integration tests validated against real GPU hardware.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

ASM="$ROOT_DIR/wave-asm/target/release/wave-asm"
EMU="$ROOT_DIR/wave-emu/target/release/wave-emu"
COMPILER="$ROOT_DIR/wave-compiler/target/release/wave-compiler"

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

check_iota_output() {
    local name="$1" file="$2" count="$3" factor="$4"
    local ok=true line=0

    while IFS= read -r val; do
        expected=$(python3 -c "print(float($line * $factor))")
        if [ "$val" != "$expected" ]; then
            fail_msg "$name" "c[$line]: expected $expected, got $val"
            ok=false
            break
        fi
        line=$((line + 1))
    done < "$file"

    if [ "$line" -ne "$count" ]; then
        fail_msg "$name" "expected $count values, got $line"
        ok=false
    fi

    if $ok; then
        pass_msg "$name"
    fi
}

check_bounds_output() {
    local name="$1" file="$2" count="$3" factor="$4" bound="$5"
    local ok=true line=0

    while IFS= read -r val; do
        if [ "$line" -lt "$bound" ]; then
            expected=$(python3 -c "print(float($line * $factor))")
        else
            expected="0.0"
        fi
        if [ "$val" != "$expected" ]; then
            fail_msg "$name" "c[$line]: expected $expected, got $val"
            ok=false
            break
        fi
        line=$((line + 1))
    done < "$file"

    if [ "$line" -ne "$count" ]; then
        fail_msg "$name" "expected $count values, got $line"
        ok=false
    fi

    if $ok; then
        pass_msg "$name"
    fi
}

write_u32_bin() {
    python3 -c "import struct; open('$1','wb').write(struct.pack('<I', $2))"
}

write_f32_bin() {
    local outfile="$1"; shift
    local values
    values=$(echo "$*" | tr ' ' ',')
    python3 -c "
import struct
data = b''
for v in [$values]:
    data += struct.pack('<f', float(v))
open('$outfile', 'wb').write(data)
"
}

run_compiler_vadd() {
    "$EMU" "$1" \
        --grid 1,1,1 \
        --fill-iota 0:f32:256:1 \
        --fill-iota 1024:f32:256:2 \
        --fill-zero 2048:f32:256 \
        --set-reg 0:0 --set-reg 1:1024 --set-reg 2:2048 --set-reg 3:"$3" \
        --dump-f32 2048:256 > "$2"
}

for crate in wave-decode wave-asm wave-dis wave-emu wave-compiler; do
    (cd "$ROOT_DIR/$crate" && cargo build --release 2>&1) | tail -1
done

"$ASM" "$ROOT_DIR/wave-metal/examples/vector_add.wave" -o "$TMPDIR/t1.wbin"
"$EMU" "$TMPDIR/t1.wbin" \
    --grid 1,1,1 --workgroup 256,1,1 \
    --fill-iota 0:f32:256:1 \
    --fill-iota 1024:f32:256:2 \
    --fill-zero 2048:f32:256 \
    --dump-f32 2048:256 > "$TMPDIR/t1.out"
check_iota_output "T01 vector_add (asm)" "$TMPDIR/t1.out" 256 3

"$COMPILER" "$ROOT_DIR/wave-compiler/examples/vector_add.py" \
    -o "$TMPDIR/t2.wbin" --lang python
run_compiler_vadd "$TMPDIR/t2.wbin" "$TMPDIR/t2.out" 256
check_iota_output "T02 vector_add (compiler/python)" "$TMPDIR/t2.out" 256 3

"$ASM" "$SCRIPT_DIR/kernels/scalar_multiply.wave" -o "$TMPDIR/t3.wbin"
"$EMU" "$TMPDIR/t3.wbin" \
    --grid 1,1,1 --workgroup 256,1,1 \
    --fill-iota 0:f32:256:1 \
    --fill-zero 1024:f32:256 \
    --dump-f32 1024:256 > "$TMPDIR/t3.out"
check_iota_output "T03 scalar_multiply (c[i]=5*i)" "$TMPDIR/t3.out" 256 5

"$ASM" "$SCRIPT_DIR/kernels/vector_sub.wave" -o "$TMPDIR/t4.wbin"
"$EMU" "$TMPDIR/t4.wbin" \
    --grid 1,1,1 --workgroup 256,1,1 \
    --fill-iota 0:f32:256:2 \
    --fill-iota 1024:f32:256:1 \
    --fill-zero 2048:f32:256 \
    --dump-f32 2048:256 > "$TMPDIR/t4.out"
check_iota_output "T04 vector_sub (c[i]=2i-i=i)" "$TMPDIR/t4.out" 256 1

"$ASM" "$SCRIPT_DIR/kernels/bounds_check.wave" -o "$TMPDIR/t5.wbin"
write_u32_bin "$TMPDIR/n200.bin" 200
"$EMU" "$TMPDIR/t5.wbin" \
    --grid 1,1,1 --workgroup 256,1,1 \
    --fill-iota 0:f32:256:1 \
    --fill-iota 1024:f32:256:2 \
    --fill-zero 2048:f32:256 \
    --arg 3072:"$TMPDIR/n200.bin" \
    --dump-f32 2048:256 > "$TMPDIR/t5.out"
check_bounds_output "T05 bounds_check (asm, n=200)" "$TMPDIR/t5.out" 256 3 200

"$COMPILER" "$ROOT_DIR/wave-compiler/examples/vector_add.py" \
    -o "$TMPDIR/t6.wbin" --lang python
run_compiler_vadd "$TMPDIR/t6.wbin" "$TMPDIR/t6.out" 256
check_iota_output "T06 vector_add (python, n=256)" "$TMPDIR/t6.out" 256 3

run_compiler_vadd "$TMPDIR/t6.wbin" "$TMPDIR/t7.out" 200
check_bounds_output "T07 vector_add (python, n=200 bounds)" "$TMPDIR/t7.out" 256 3 200

"$COMPILER" "$ROOT_DIR/wave-compiler/examples/vector_add.rs" \
    -o "$TMPDIR/t8.wbin" --lang rust
run_compiler_vadd "$TMPDIR/t8.wbin" "$TMPDIR/t8.out" 256
check_iota_output "T08 vector_add (rust)" "$TMPDIR/t8.out" 256 3

"$COMPILER" "$ROOT_DIR/wave-compiler/examples/vector_add.cpp" \
    -o "$TMPDIR/t9.wbin" --lang cpp
run_compiler_vadd "$TMPDIR/t9.wbin" "$TMPDIR/t9.out" 256
check_iota_output "T09 vector_add (c++)" "$TMPDIR/t9.out" 256 3

"$COMPILER" "$ROOT_DIR/wave-compiler/examples/vector_add.ts" \
    -o "$TMPDIR/t10.wbin" --lang typescript
run_compiler_vadd "$TMPDIR/t10.wbin" "$TMPDIR/t10.out" 256
check_iota_output "T10 vector_add (typescript)" "$TMPDIR/t10.out" 256 3

cat > "$TMPDIR/sum_reduce.py" << 'PYEOF'
@kernel
def sum_reduce(a: f32[:], out: f32[:], n: u32):
    gid = thread_id()
    if gid == 0:
        total: f32 = 0.0
        for i in range(n):
            total = total + a[i]
        out[0] = total
PYEOF
"$COMPILER" "$TMPDIR/sum_reduce.py" -o "$TMPDIR/t11.wbin" --lang python
write_f32_bin "$TMPDIR/t11_input.bin" 1 2 3 4 5 6 7 8
"$EMU" "$TMPDIR/t11.wbin" \
    --grid 1,1,1 --workgroup 256,1,1 \
    --fill-zero 0:f32:256 \
    --arg 0:"$TMPDIR/t11_input.bin" \
    --fill-zero 1024:f32:256 \
    --set-reg 0:0 --set-reg 1:1024 --set-reg 2:8 \
    --dump-f32 1024:1 > "$TMPDIR/t11.out"
t11_val=$(cat "$TMPDIR/t11.out")
if [ "$t11_val" = "36.0" ]; then
    pass_msg "T11 sum_reduce (loop, sum(1..8)=36)"
else
    fail_msg "T11 sum_reduce (loop, sum(1..8)=36)" "expected 36.0, got $t11_val"
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"
exit "$FAIL"
