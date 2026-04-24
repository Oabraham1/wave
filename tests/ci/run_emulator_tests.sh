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

write_f16_bin() {
    local outfile="$1"; shift
    local values
    values=$(echo "$*" | tr ' ' ',')
    python3 -c "
import struct
def f32_to_f16(val):
    bits = struct.unpack('<I', struct.pack('<f', val))[0]
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    if exp == 0:
        return sign << 15
    elif exp == 0xFF:
        return (sign << 15) | 0x7C00 | (mant >> 13)
    new_exp = exp - 127 + 15
    if new_exp <= 0:
        return sign << 15
    elif new_exp >= 31:
        return (sign << 15) | 0x7C00
    return (sign << 15) | (new_exp << 10) | (mant >> 13)
data = b''
for v in [$values]:
    data += struct.pack('<H', f32_to_f16(float(v)))
open('$outfile', 'wb').write(data)
"
}

write_bf16_bin() {
    local outfile="$1"; shift
    local values
    values=$(echo "$*" | tr ' ' ',')
    python3 -c "
import struct
data = b''
for v in [$values]:
    bits = struct.unpack('<I', struct.pack('<f', float(v)))[0]
    data += struct.pack('<H', bits >> 16)
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

"$ASM" "$SCRIPT_DIR/kernels/gemm_blocked.wave" -o "$TMPDIR/t12.wbin"

N=128
A_SIZE=$((N * N * 4))
B_OFF=$A_SIZE
C_OFF=$((B_OFF + A_SIZE))
MEM=$((C_OFF + A_SIZE + 4096))

python3 -c "
import struct, random
random.seed(42)
N = $N
A = [random.uniform(-1, 1) for _ in range(N * N)]
B = [random.uniform(-1, 1) for _ in range(N * N)]
fmt = '<' + str(N * N) + 'f'
open('$TMPDIR/t12_a.bin', 'wb').write(struct.pack(fmt, *A))
open('$TMPDIR/t12_b.bin', 'wb').write(struct.pack(fmt, *B))
"

"$EMU" "$TMPDIR/t12.wbin" \
    --grid 1,1,1 --workgroup 16,16,1 \
    --device-memory $MEM \
    --fill-zero 0:f32:$((N * N)) --arg 0:"$TMPDIR/t12_a.bin" \
    --fill-zero ${B_OFF}:f32:$((N * N)) --arg ${B_OFF}:"$TMPDIR/t12_b.bin" \
    --fill-zero ${C_OFF}:f32:$((N * N)) \
    --set-reg 0:0 --set-reg 1:${B_OFF} --set-reg 2:${C_OFF} --set-reg 3:${N} --set-reg 4:${N} --set-reg 5:${N} \
    --dump-f32 ${C_OFF}:$((N * N)) > "$TMPDIR/t12.out"

t12_ok=$(python3 -c "
import random
random.seed(42)
N = $N
A = [random.uniform(-1, 1) for _ in range(N * N)]
B = [random.uniform(-1, 1) for _ in range(N * N)]
C_ref = [0.0] * (N * N)
for i in range(N):
    for j in range(N):
        for k in range(N):
            C_ref[i * N + j] += A[i * N + k] * B[k * N + j]
with open('$TMPDIR/t12.out') as f:
    values = [float(line) for line in f]
max_err = max(abs(values[i] - C_ref[i]) for i in range(N * N))
print('ok' if max_err < 1e-4 else f'max_err={max_err:.2e}')
")

if [ "$t12_ok" = "ok" ]; then
    pass_msg "T12 gemm_register_blocked_8x8 (128x128, random)"
else
    fail_msg "T12 gemm_register_blocked_8x8 (128x128, random)" "$t12_ok"
fi

"$ASM" "$SCRIPT_DIR/kernels/vector_add_f16.wave" -o "$TMPDIR/t13.wbin"
write_f16_bin "$TMPDIR/t13_a.bin" 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
write_f16_bin "$TMPDIR/t13_b.bin" 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5
"$EMU" "$TMPDIR/t13.wbin" \
    --grid 1,1,1 --workgroup 256,1,1 \
    --device-memory 4096 \
    --arg 0:"$TMPDIR/t13_a.bin" --arg 16:"$TMPDIR/t13_b.bin" \
    --fill-zero 32:u8:16 \
    --set-reg 1:0 --set-reg 2:16 --set-reg 3:32 \
    --dump-f16 32:8 > "$TMPDIR/t13.out"

t13_ok=$(python3 -c "
expected = [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5]
with open('$TMPDIR/t13.out') as f:
    values = [float(line) for line in f]
ok = len(values) == 8 and all(abs(values[i] - expected[i]) < 0.01 for i in range(8))
print('ok' if ok else f'got {values}')
")

if [ "$t13_ok" = "ok" ]; then
    pass_msg "T13 vector_add_f16 (f16 vadd, 8 elements)"
else
    fail_msg "T13 vector_add_f16 (f16 vadd, 8 elements)" "$t13_ok"
fi

"$ASM" "$SCRIPT_DIR/kernels/vector_add_bf16.wave" -o "$TMPDIR/t14.wbin"
write_bf16_bin "$TMPDIR/t14_a.bin" 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
write_bf16_bin "$TMPDIR/t14_b.bin" 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5
"$EMU" "$TMPDIR/t14.wbin" \
    --grid 1,1,1 --workgroup 256,1,1 \
    --device-memory 4096 \
    --arg 0:"$TMPDIR/t14_a.bin" --arg 16:"$TMPDIR/t14_b.bin" \
    --fill-zero 32:u8:16 \
    --set-reg 1:0 --set-reg 2:16 --set-reg 3:32 \
    --dump-bf16 32:8 > "$TMPDIR/t14.out"

t14_ok=$(python3 -c "
expected = [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5]
with open('$TMPDIR/t14.out') as f:
    values = [float(line) for line in f]
ok = len(values) == 8 and all(abs(values[i] - expected[i]) < 0.1 for i in range(8))
print('ok' if ok else f'got {values}')
")

if [ "$t14_ok" = "ok" ]; then
    pass_msg "T14 vector_add_bf16 (bf16 vadd, 8 elements)"
else
    fail_msg "T14 vector_add_bf16 (bf16 vadd, 8 elements)" "$t14_ok"
fi

"$ASM" "$SCRIPT_DIR/kernels/gemm_blocked_f16.wave" -o "$TMPDIR/t15.wbin"

N=128
A_SIZE=$((N * N * 2))
B_OFF=$A_SIZE
C_OFF=$((B_OFF + A_SIZE))
MEM=$((C_OFF + A_SIZE + 4096))

write_f16_bin "$TMPDIR/t15_a.bin" $(python3 -c "
import random
random.seed(42)
print(' '.join(str(random.uniform(-1, 1)) for _ in range($N * $N)))
")

write_f16_bin "$TMPDIR/t15_b.bin" $(python3 -c "
import random
random.seed(42)
[random.uniform(-1, 1) for _ in range($N * $N)]
print(' '.join(str(random.uniform(-1, 1)) for _ in range($N * $N)))
")

"$EMU" "$TMPDIR/t15.wbin" \
    --grid 1,1,1 --workgroup 16,16,1 \
    --device-memory $MEM \
    --arg 0:"$TMPDIR/t15_a.bin" --arg ${B_OFF}:"$TMPDIR/t15_b.bin" \
    --set-reg 0:0 --set-reg 1:${B_OFF} --set-reg 2:${C_OFF} --set-reg 3:${N} --set-reg 4:${N} --set-reg 5:${N} \
    --dump-f16 ${C_OFF}:$((N * N)) > "$TMPDIR/t15.out"

t15_ok=$(python3 -c "
import struct, random
random.seed(42)
N = $N
def f32_to_f16(val):
    bits = struct.unpack('<I', struct.pack('<f', val))[0]
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    if exp == 0: return sign << 15
    elif exp == 0xFF: return (sign << 15) | 0x7C00 | (mant >> 13)
    new_exp = exp - 127 + 15
    if new_exp <= 0: return sign << 15
    elif new_exp >= 31: return (sign << 15) | 0x7C00
    return (sign << 15) | (new_exp << 10) | (mant >> 13)
def f16_to_f32(h):
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0: return (-1)**sign * 2**(-14) * (mant / 1024.0) if mant else 0.0
    elif exp == 31: return float('inf') if mant == 0 else float('nan')
    return (-1)**sign * 2**(exp - 15) * (1 + mant / 1024.0)
A = [random.uniform(-1, 1) for _ in range(N * N)]
B = [random.uniform(-1, 1) for _ in range(N * N)]
Ah = [f16_to_f32(f32_to_f16(v)) for v in A]
Bh = [f16_to_f32(f32_to_f16(v)) for v in B]
C_ref = [0.0] * (N * N)
for i in range(N):
    for j in range(N):
        for k in range(N):
            C_ref[i * N + j] += Ah[i * N + k] * Bh[k * N + j]
with open('$TMPDIR/t15.out') as f:
    values = [float(line) for line in f]
max_err = max(abs(values[i] - C_ref[i]) for i in range(N * N))
print('ok' if max_err < 0.01 else f'max_err={max_err:.2e}')
")

if [ "$t15_ok" = "ok" ]; then
    pass_msg "T15 gemm_blocked_f16 (mixed-precision, 128x128)"
else
    fail_msg "T15 gemm_blocked_f16 (mixed-precision, 128x128)" "$t15_ok"
fi

"$ASM" "$SCRIPT_DIR/kernels/mma_query.wave" -o "$TMPDIR/t16.wbin"
"$EMU" "$TMPDIR/t16.wbin" --grid 1,1,1 --workgroup 1,1,1 \
    --device-memory 256 --fill-zero 0:f32:64 \
    --dump-u32 0:4 > "$TMPDIR/t16.out"

t16_ok=$(python3 -c "
with open('$TMPDIR/t16.out') as f:
    values = [int(float(line)) for line in f]
expected = [1, 4, 4, 4]
ok = values == expected
print('ok' if ok else f'got {values}')
")

if [ "$t16_ok" = "ok" ]; then
    pass_msg "T16 mma_query (sr_mma_supported=1, m=4, n=4, k=4)"
else
    fail_msg "T16 mma_query (sr_mma_supported=1, m=4, n=4, k=4)" "$t16_ok"
fi

"$ASM" "$SCRIPT_DIR/kernels/mma_simple.wave" -o "$TMPDIR/t17.wbin"

python3 -c "
import struct
I = [1.0 if i==j else 0.0 for i in range(4) for j in range(4)]
data = struct.pack('<16f', *I)
open('$TMPDIR/t17_i.bin', 'wb').write(data)
"

"$EMU" "$TMPDIR/t17.wbin" --grid 1,1,1 --workgroup 1,1,1 \
    --device-memory 256 --fill-zero 0:f32:64 \
    --arg 0:"$TMPDIR/t17_i.bin" --arg 64:"$TMPDIR/t17_i.bin" \
    --dump-f32 128:16 > "$TMPDIR/t17.out"

t17_ok=$(python3 -c "
with open('$TMPDIR/t17.out') as f:
    values = [float(line) for line in f]
expected = [1.0 if i==j else 0.0 for i in range(4) for j in range(4)]
ok = len(values) == 16 and all(abs(values[i] - expected[i]) < 1e-6 for i in range(16))
print('ok' if ok else f'got {values[:4]}...')
")

if [ "$t17_ok" = "ok" ]; then
    pass_msg "T17 mma_simple (identity * identity = identity)"
else
    fail_msg "T17 mma_simple (identity * identity = identity)" "$t17_ok"
fi

"$ASM" "$SCRIPT_DIR/kernels/mma_simple.wave" -o "$TMPDIR/t18.wbin"

python3 -c "
import struct
data = struct.pack('<16f', *([1.0]*16))
open('$TMPDIR/t18_ones.bin', 'wb').write(data)
"

"$EMU" "$TMPDIR/t18.wbin" --grid 1,1,1 --workgroup 1,1,1 \
    --device-memory 256 --fill-zero 0:f32:64 \
    --arg 0:"$TMPDIR/t18_ones.bin" --arg 64:"$TMPDIR/t18_ones.bin" \
    --dump-f32 128:16 > "$TMPDIR/t18.out"

t18_ok=$(python3 -c "
with open('$TMPDIR/t18.out') as f:
    values = [float(line) for line in f]
ok = len(values) == 16 and all(abs(v - 4.0) < 1e-6 for v in values)
print('ok' if ok else f'got {values[:4]}...')
")

if [ "$t18_ok" = "ok" ]; then
    pass_msg "T18 mma_ones (ones * ones = all 4.0)"
else
    fail_msg "T18 mma_ones (ones * ones = all 4.0)" "$t18_ok"
fi

"$ASM" "$SCRIPT_DIR/kernels/gemm_bias_relu.wave" -o "$TMPDIR/t19.wbin"

N=128
A_SIZE=$((N * N * 4))
B_OFF=$A_SIZE
BIAS_OFF=$((B_OFF + A_SIZE))
C_OFF=$((BIAS_OFF + N * 4))
MEM=$((C_OFF + A_SIZE + 4096))

python3 -c "
import struct
N = $N
I = [1.0 if i==j else 0.0 for i in range(N) for j in range(N)]
B = [(i - 64) * 0.1 for i in range(N) for j in range(N)]
bias = [0.0] * N
fmt = '<' + str(N*N) + 'f'
bfmt = '<' + str(N) + 'f'
open('$TMPDIR/t19_a.bin', 'wb').write(struct.pack(fmt, *I))
open('$TMPDIR/t19_b.bin', 'wb').write(struct.pack(fmt, *B))
open('$TMPDIR/t19_bias.bin', 'wb').write(struct.pack(bfmt, *bias))
"

"$EMU" "$TMPDIR/t19.wbin" \
    --grid 1,1,1 --workgroup 16,16,1 \
    --device-memory $MEM \
    --arg 0:"$TMPDIR/t19_a.bin" --arg ${B_OFF}:"$TMPDIR/t19_b.bin" --arg ${BIAS_OFF}:"$TMPDIR/t19_bias.bin" \
    --fill-zero ${C_OFF}:f32:$((N * N)) \
    --set-reg 0:0 --set-reg 1:${B_OFF} --set-reg 2:${BIAS_OFF} --set-reg 3:${C_OFF} --set-reg 4:${N} --set-reg 5:${N} \
    --dump-f32 ${C_OFF}:$((N * N)) > "$TMPDIR/t19.out"

t19_ok=$(python3 -c "
N = $N
with open('$TMPDIR/t19.out') as f:
    values = [float(line) for line in f]
if len(values) != N * N:
    print(f'wrong count: {len(values)}')
else:
    ok = True
    for i in range(N):
        for j in range(N):
            expected = max(0.0, (i - 64) * 0.1)
            actual = values[i * N + j]
            if abs(actual - expected) > 1e-4:
                print(f'mismatch [{i}][{j}]: expected {expected}, got {actual}')
                ok = False
                break
        if not ok:
            break
    if ok:
        print('ok')
")

if [ "$t19_ok" = "ok" ]; then
    pass_msg "T19 gemm_bias_relu (identity, 128x128)"
else
    fail_msg "T19 gemm_bias_relu (identity, 128x128)" "$t19_ok"
fi

"$ASM" "$SCRIPT_DIR/kernels/gemm_bias_relu.wave" -o "$TMPDIR/t20.wbin"

N=256
A_SIZE=$((N * N * 4))
B_OFF=$A_SIZE
BIAS_OFF=$((B_OFF + A_SIZE))
C_OFF=$((BIAS_OFF + N * 4))
MEM=$((C_OFF + A_SIZE + 4096))

python3 -c "
import struct, random
random.seed(42)
N = $N
A = [random.uniform(-1, 1) for _ in range(N * N)]
B = [random.uniform(-1, 1) for _ in range(N * N)]
bias = [0.5] * N
fmt = '<' + str(N * N) + 'f'
bfmt = '<' + str(N) + 'f'
open('$TMPDIR/t20_a.bin', 'wb').write(struct.pack(fmt, *A))
open('$TMPDIR/t20_b.bin', 'wb').write(struct.pack(fmt, *B))
open('$TMPDIR/t20_bias.bin', 'wb').write(struct.pack(bfmt, *bias))
"

GRID_X=$((N / 128))
GRID_Y=$((N / 128))

"$EMU" "$TMPDIR/t20.wbin" \
    --grid ${GRID_X},${GRID_Y},1 --workgroup 16,16,1 \
    --device-memory $MEM \
    --arg 0:"$TMPDIR/t20_a.bin" --arg ${B_OFF}:"$TMPDIR/t20_b.bin" --arg ${BIAS_OFF}:"$TMPDIR/t20_bias.bin" \
    --fill-zero ${C_OFF}:f32:$((N * N)) \
    --set-reg 0:0 --set-reg 1:${B_OFF} --set-reg 2:${BIAS_OFF} --set-reg 3:${C_OFF} --set-reg 4:${N} --set-reg 5:${N} \
    --dump-f32 ${C_OFF}:$((N * N)) > "$TMPDIR/t20.out"

t20_ok=$(python3 -c "
import random
random.seed(42)
N = $N
A = [random.uniform(-1, 1) for _ in range(N * N)]
B = [random.uniform(-1, 1) for _ in range(N * N)]
C_ref = [0.0] * (N * N)
for i in range(N):
    for j in range(N):
        s = 0.0
        for k in range(N):
            s += A[i * N + k] * B[k * N + j]
        C_ref[i * N + j] = max(0.0, s + 0.5)
with open('$TMPDIR/t20.out') as f:
    values = [float(line) for line in f]
if len(values) != N * N:
    print(f'wrong count: {len(values)}')
else:
    max_err = max(abs(values[i] - C_ref[i]) for i in range(N * N))
    print('ok' if max_err < 1e-3 else f'max_abs_err={max_err:.2e}')
")

if [ "$t20_ok" = "ok" ]; then
    pass_msg "T20 gemm_bias_relu (random, 256x256, bias=0.5)"
else
    fail_msg "T20 gemm_bias_relu (random, 256x256, bias=0.5)" "$t20_ok"
fi

"$ASM" "$SCRIPT_DIR/kernels/gemm_bias_relu_f16.wave" -o "$TMPDIR/t21.wbin"

N=128
A_SIZE=$((N * N * 2))
B_OFF=$A_SIZE
BIAS_OFF=$((B_OFF + A_SIZE))
C_OFF=$((BIAS_OFF + N * 4))
MEM=$((C_OFF + N * N * 4 + 4096))

python3 -c "
import struct, random
random.seed(99)
N = $N
def f32_to_f16(val):
    bits = struct.unpack('<I', struct.pack('<f', val))[0]
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    if exp == 0: return sign << 15
    elif exp == 0xFF: return (sign << 15) | 0x7C00 | (mant >> 13)
    new_exp = exp - 127 + 15
    if new_exp <= 0: return sign << 15
    elif new_exp >= 31: return (sign << 15) | 0x7C00
    return (sign << 15) | (new_exp << 10) | (mant >> 13)
A = [random.uniform(-1, 1) for _ in range(N * N)]
B = [random.uniform(-1, 1) for _ in range(N * N)]
bias = [1.0] * N
ah = b''
bh = b''
for v in A:
    ah += struct.pack('<H', f32_to_f16(v))
for v in B:
    bh += struct.pack('<H', f32_to_f16(v))
open('$TMPDIR/t21_a.bin', 'wb').write(ah)
open('$TMPDIR/t21_b.bin', 'wb').write(bh)
open('$TMPDIR/t21_bias.bin', 'wb').write(struct.pack('<' + str(N) + 'f', *bias))
"

"$EMU" "$TMPDIR/t21.wbin" \
    --grid 1,1,1 --workgroup 16,16,1 \
    --device-memory $MEM \
    --arg 0:"$TMPDIR/t21_a.bin" --arg ${B_OFF}:"$TMPDIR/t21_b.bin" --arg ${BIAS_OFF}:"$TMPDIR/t21_bias.bin" \
    --fill-zero ${C_OFF}:f32:$((N * N)) \
    --set-reg 0:0 --set-reg 1:${B_OFF} --set-reg 2:${BIAS_OFF} --set-reg 3:${C_OFF} --set-reg 4:${N} --set-reg 5:${N} \
    --dump-f16 ${C_OFF}:$((N * N)) > "$TMPDIR/t21.out"

t21_ok=$(python3 -c "
import struct, random
random.seed(99)
N = $N
def f32_to_f16(val):
    bits = struct.unpack('<I', struct.pack('<f', val))[0]
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    if exp == 0: return sign << 15
    elif exp == 0xFF: return (sign << 15) | 0x7C00 | (mant >> 13)
    new_exp = exp - 127 + 15
    if new_exp <= 0: return sign << 15
    elif new_exp >= 31: return (sign << 15) | 0x7C00
    return (sign << 15) | (new_exp << 10) | (mant >> 13)
def f16_to_f32(h):
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0: return (-1)**sign * 2**(-14) * (mant / 1024.0) if mant else 0.0
    elif exp == 31: return float('inf') if mant == 0 else float('nan')
    return (-1)**sign * 2**(exp - 15) * (1 + mant / 1024.0)
A = [random.uniform(-1, 1) for _ in range(N * N)]
B = [random.uniform(-1, 1) for _ in range(N * N)]
Ah = [f16_to_f32(f32_to_f16(v)) for v in A]
Bh = [f16_to_f32(f32_to_f16(v)) for v in B]
C_ref = [0.0] * (N * N)
for i in range(N):
    for j in range(N):
        s = 0.0
        for k in range(N):
            s += Ah[i * N + k] * Bh[k * N + j]
        C_ref[i * N + j] = max(0.0, s + 1.0)
with open('$TMPDIR/t21.out') as f:
    values = [float(line) for line in f]
if len(values) != N * N:
    print(f'wrong count: {len(values)}')
else:
    max_err = 0.0
    for i in range(N * N):
        ref = C_ref[i]
        diff = abs(values[i] - ref)
        rel = diff / max(abs(ref), 1e-6)
        if rel > max_err:
            max_err = rel
    print('ok' if max_err < 0.01 else f'max_rel_err={max_err:.2e}')
")

if [ "$t21_ok" = "ok" ]; then
    pass_msg "T21 gemm_bias_relu_f16 (mixed-precision, 128x128)"
else
    fail_msg "T21 gemm_bias_relu_f16 (mixed-precision, 128x128)" "$t21_ok"
fi

extract_device_ops() {
    local stats_output="$1"
    local loads stores
    loads=$(echo "$stats_output" | grep "Loads:" | head -1 | sed 's/.*Loads:[[:space:]]*//' | sed 's/[[:space:]].*//')
    stores=$(echo "$stats_output" | grep "Stores:" | head -1 | sed 's/.*Stores:[[:space:]]*//' | sed 's/[[:space:]].*//')
    echo $((loads + stores))
}

"$ASM" "$SCRIPT_DIR/kernels/gemm_blocked.wave" -o "$TMPDIR/t22_gemm.wbin"
"$ASM" "$SCRIPT_DIR/kernels/bias_add.wave" -o "$TMPDIR/t22_bias.wbin"
"$ASM" "$SCRIPT_DIR/kernels/relu.wave" -o "$TMPDIR/t22_relu.wbin"
"$ASM" "$SCRIPT_DIR/kernels/gemm_bias_relu.wave" -o "$TMPDIR/t22_fused.wbin"

N=128
A_SIZE=$((N * N * 4))
B_OFF=$A_SIZE
BIAS_OFF=$((B_OFF + A_SIZE))
C_OFF=$((BIAS_OFF + N * 4))
MEM=$((C_OFF + A_SIZE + 4096))

python3 -c "
import struct, random
random.seed(42)
N = $N
A = [random.uniform(-1, 1) for _ in range(N * N)]
B = [random.uniform(-1, 1) for _ in range(N * N)]
bias = [0.5] * N
fmt = '<' + str(N * N) + 'f'
bfmt = '<' + str(N) + 'f'
open('$TMPDIR/t22_a.bin', 'wb').write(struct.pack(fmt, *A))
open('$TMPDIR/t22_b.bin', 'wb').write(struct.pack(fmt, *B))
open('$TMPDIR/t22_bias.bin', 'wb').write(struct.pack(bfmt, *bias))
open('$TMPDIR/t22_dummy.bin', 'wb').write(struct.pack(fmt, *([0.0] * (N * N))))
"

GEMM_STATS=$("$EMU" "$TMPDIR/t22_gemm.wbin" \
    --grid 1,1,1 --workgroup 16,16,1 \
    --device-memory $MEM \
    --arg 0:"$TMPDIR/t22_a.bin" --arg ${B_OFF}:"$TMPDIR/t22_b.bin" \
    --fill-zero ${C_OFF}:f32:$((N * N)) \
    --set-reg 0:0 --set-reg 1:${B_OFF} --set-reg 2:${C_OFF} --set-reg 3:${N} --set-reg 4:${N} --set-reg 5:${N} \
    --stats 2>&1)

TOTAL_ELEMS=$((N * N))
NUM_WG=$(( (TOTAL_ELEMS + 255) / 256 ))

BIAS_STATS=$("$EMU" "$TMPDIR/t22_bias.wbin" \
    --grid ${NUM_WG},1,1 --workgroup 256,1,1 \
    --device-memory $MEM \
    --arg 0:"$TMPDIR/t22_dummy.bin" --arg ${A_SIZE}:"$TMPDIR/t22_bias.bin" \
    --fill-zero ${B_OFF}:f32:$((N * N)) \
    --set-reg 0:0 --set-reg 1:${A_SIZE} --set-reg 2:${B_OFF} --set-reg 3:${N} --set-reg 4:${TOTAL_ELEMS} \
    --stats 2>&1)

RELU_STATS=$("$EMU" "$TMPDIR/t22_relu.wbin" \
    --grid ${NUM_WG},1,1 --workgroup 256,1,1 \
    --device-memory $MEM \
    --arg 0:"$TMPDIR/t22_dummy.bin" \
    --fill-zero ${A_SIZE}:f32:$((N * N)) \
    --set-reg 0:0 --set-reg 1:${A_SIZE} --set-reg 2:${TOTAL_ELEMS} \
    --stats 2>&1)

GEMM_OPS=$(extract_device_ops "$GEMM_STATS")
BIAS_OPS=$(extract_device_ops "$BIAS_STATS")
RELU_OPS=$(extract_device_ops "$RELU_STATS")
UNFUSED_TOTAL=$((GEMM_OPS + BIAS_OPS + RELU_OPS))

FUSED_STATS=$("$EMU" "$TMPDIR/t22_fused.wbin" \
    --grid 1,1,1 --workgroup 16,16,1 \
    --device-memory $MEM \
    --arg 0:"$TMPDIR/t22_a.bin" --arg ${B_OFF}:"$TMPDIR/t22_b.bin" --arg ${BIAS_OFF}:"$TMPDIR/t22_bias.bin" \
    --fill-zero ${C_OFF}:f32:$((N * N)) \
    --set-reg 0:0 --set-reg 1:${B_OFF} --set-reg 2:${BIAS_OFF} --set-reg 3:${C_OFF} --set-reg 4:${N} --set-reg 5:${N} \
    --stats 2>&1)

FUSED_OPS=$(extract_device_ops "$FUSED_STATS")

echo "  Unfused device ops: GEMM=$GEMM_OPS + bias=$BIAS_OPS + relu=$RELU_OPS = $UNFUSED_TOTAL"
echo "  Fused device ops:   $FUSED_OPS"

if [ "$FUSED_OPS" -lt "$UNFUSED_TOTAL" ]; then
    pass_msg "T22 fusion memory savings (fused=$FUSED_OPS < unfused=$UNFUSED_TOTAL)"
else
    fail_msg "T22 fusion memory savings" "fused=$FUSED_OPS >= unfused=$UNFUSED_TOTAL"
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"
exit "$FAIL"
