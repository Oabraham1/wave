"""Literal f32 witness for separate operations and explicit fused multiply-add."""

import struct

NAME = "rounding_boundaries"
INITIAL_WORDS = [0x3F800001, 0x3F7FFFFE, 0xBF800000, 0x3F800000,
                 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF]
EXPECTED_WORDS = INITIAL_WORDS[:4] + [0x00000000, 0x00000000, 0xA8800000, 0xDEADBEEF]
CONTRACT = {
    "name": NAME, "groups": 1, "threads": 1, "dispatches": 1,
    "initial_path": NAME + ".input.bin", "expected_path": NAME + ".expected.bin",
    "wbin_path": NAME + ".wbin", "ptx_path": NAME + ".wave.ptx",
    "initial_words": INITIAL_WORDS, "expected_words": EXPECTED_WORDS,
    "output_word_offset": 4, "output_words": 3,
    "operations": ["fmul followed by fadd", "fmul followed by fsub", "explicit fma"],
    "arithmetic": "a*b is exactly 1 - 2^-46; separate multiplication rounds to 1; explicit fma retains -2^-46",
    "purpose": "required correctness check before workload timing; excluded from timing summaries",
}
SOURCE = """.kernel rounding_boundaries
.registers 16
.workgroup_size 1, 1, 1
.local_memory 0
mov_imm r0, 0
device_load_u32 r1, r0
mov_imm r0, 4
device_load_u32 r2, r0
mov_imm r0, 8
device_load_u32 r3, r0
mov_imm r0, 12
device_load_u32 r4, r0
fmul r5, r1, r2
fadd r6, r5, r3
fsub r7, r5, r4
fma r8, r1, r2, r3
mov_imm r0, 16
device_store_u32 r0, r6
mov_imm r0, 20
device_store_u32 r0, r7
mov_imm r0, 24
device_store_u32 r0, r8
halt
.end
"""


def initial_bytes():
    return struct.pack("<8I", *INITIAL_WORDS)


def expected_bytes():
    return struct.pack("<8I", *EXPECTED_WORDS)


def validate_contract(spec):
    for key, value in CONTRACT.items():
        if spec.get(key) != value:
            raise ValueError(f"rounding witness contract differs: {key}")
