"""Construct matched WAVE and Metal workloads with deterministic inputs."""

from array import array
import math
import random
import struct


WORKLOADS = (
    "vector_add", "divergent", "gather", "histogram_uniform", "histogram_skewed",
    "reduce_shuffle", "reduce_shared",
)
APPLICATION_WORKLOADS = ("csr_spmv", "stencil_2d")
ALL_WORKLOADS = WORKLOADS + APPLICATION_WORKLOADS
THREADS = 128
WAVE_WIDTH = 32
GUARD = 0xA5C39E71


def float_word(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def assembly(name):
    if name in APPLICATION_WORKLOADS:
        return application_assembly(name)
    lines = [
        f".kernel {name}", ".registers 32", f".workgroup_size {THREADS}, 1, 1",
        ".local_memory 512", "mov_imm r4, 0", "device_load_u32 r3, r4",
        "mov_imm r4, 4", "device_load_u32 r0, r4", "mov_imm r4, 8",
        "device_load_u32 r1, r4", "mov_imm r4, 12", "device_load_u32 r2, r4",
        "mov_sr r5, sr_thread_id_x", "mov_sr r6, sr_workgroup_id_x",
        f"mov_imm r7, {THREADS}", "imul r8, r6, r7", "iadd r8, r8, r5",
        "mov_imm r7, 4", "imul r9, r8, r7", "iadd r10, r9, r0",
        "ucmp_lt p1, r8, r3",
    ]
    if name.startswith("reduce_"):
        lines += ["mov_imm r11, 0", "if p1", "device_load_u32 r11, r10", "endif"]
        if name == "reduce_shuffle":
            for offset in (16, 8, 4, 2, 1):
                lines += [f"mov_imm r12, {offset}", "wave_shuffle_xor r13, r11, r12",
                          "fadd r11, r11, r13"]
        else:
            lines += ["imul r14, r5, r7"]
            for offset in (16, 8, 4, 2, 1):
                lines += ["local_store_u32 r14, r11", "barrier",
                          f"mov_imm r12, {offset}", "xor r13, r5, r12",
                          "imul r13, r13, r7", "local_load_u32 r15, r13",
                          "barrier", "fadd r11, r11, r15"]
        lines += ["mov_sr r16, sr_lane_id", "mov_imm r17, 0", "icmp_eq p2, r16, r17",
                  "if p2", "mov_imm r18, 5", "shr r19, r8, r18",
                  "imul r19, r19, r7", "iadd r19, r19, r2",
                  "device_store_u32 r19, r11", "endif"]
    else:
        lines += ["if p1", "device_load_u32 r11, r10"]
        if name == "vector_add":
            lines += ["iadd r10, r9, r1", "device_load_u32 r12, r10", "fadd r13, r11, r12"]
        elif name == "divergent":
            lines += ["mov_imm r12, 0", "fcmp_lt p2, r11, r12", "if p2",
                      "mov_imm r12, 1065353216", "fsub r13, r11, r12", "else",
                      "mov_imm r12, 1073741824", "fmul r13, r11, r12", "endif"]
        elif name == "gather":
            lines += ["iadd r10, r9, r1", "device_load_u32 r12, r10",
                      "imul r12, r12, r7", "iadd r12, r12, r0", "device_load_u32 r13, r12"]
        elif name.startswith("histogram_"):
            lines += ["imul r12, r11, r7", "iadd r12, r12, r2", "mov_imm r13, 1",
                      "atomic_add r14, r12, r13, .device"]
        if not name.startswith("histogram_"):
            lines += ["iadd r10, r9, r2", "device_store_u32 r10, r13"]
        lines += ["endif"]
    return "\n".join(lines + ["halt", ".end", ""])


def native_source(name):
    if name in APPLICATION_WORKLOADS:
        return application_native_source(name)
    prefix = f"""#include <metal_stdlib>
using namespace metal;
kernel void {name}(
    device uint* memory [[buffer(0)]],
    uint i [[thread_position_in_grid]],
    uint t [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {{
    const uint n = memory[0];
    device float* a = (device float*)((device uchar*)memory + memory[1]);
    device uint* b = (device uint*)((device uchar*)memory + memory[2]);
    device float* output = (device float*)((device uchar*)memory + memory[3]);
"""
    if name == "vector_add":
        body = "    if (i < n) output[i] = a[i] + as_type<float>(b[i]);\n"
    elif name == "divergent":
        body = "    if (i < n) { float x = a[i]; output[i] = x < 0.0f ? x - 1.0f : x * 2.0f; }\n"
    elif name == "gather":
        body = "    if (i < n) output[i] = a[b[i]];\n"
    elif name.startswith("histogram_"):
        body = "    if (i < n) atomic_fetch_add_explicit((device atomic_uint*)output + as_type<uint>(a[i]), 1u, memory_order_relaxed);\n"
    else:
        body = "    float value = i < n ? a[i] : 0.0f;\n"
        if name == "reduce_shared":
            body += f"    threadgroup float scratch[{THREADS}];\n"
        for offset in (16, 8, 4, 2, 1):
            if name == "reduce_shuffle":
                body += f"    value += simd_shuffle_xor(value, {offset});\n"
            else:
                body += f"""    scratch[t] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float other{offset} = scratch[t ^ {offset}];
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    value += other{offset};
"""
        body += "    if (lane == 0) output[i / 32] = value;\n"
    return prefix + body + "}\n"


def make_case(name, size, seed):
    if name in APPLICATION_WORKLOADS:
        return make_application_case(name, size, seed)
    rng = random.Random(seed)
    a = [float(((i * 37 + 11) % 251) - 125) / 16 for i in range(size)]
    b = [float(((i * 17 + 3) % 127) - 63) / 8 for i in range(size)]
    a_words = [float_word(x) for x in a]
    b_words = [float_word(x) for x in b]
    if name == "vector_add":
        expected = [float_word(x + y) for x, y in zip(a, b)]
    elif name == "divergent":
        expected = [float_word(x - 1 if x < 0 else x * 2) for x in a]
    elif name == "gather":
        b_words = list(range(size))
        rng.shuffle(b_words)
        expected = [a_words[index] for index in b_words]
    elif name.startswith("histogram_"):
        a_words = [rng.randrange(64) for _ in range(size)]
        if name == "histogram_skewed":
            a_words = [0 if rng.randrange(10) < 9 else x for x in a_words]
        expected = [0] * 64
        for x in a_words:
            expected[x] += 1
    else:
        padded_size = math.ceil(size / THREADS) * THREADS
        padded = a + [0.0] * (padded_size - size)
        expected = [float_word(sum(padded[i:i + WAVE_WIDTH]))
                    for i in range(0, padded_size, WAVE_WIDTH)]
    # Guard words separate every region and bound the complete allocation.
    a_offset = 32
    b_offset = a_offset + size + 16
    out_offset = b_offset + size + 16
    words = [GUARD] * (out_offset + len(expected) + 16)
    words[:4] = [size, a_offset * 4, b_offset * 4, out_offset * 4]
    words[a_offset:a_offset + size] = a_words
    words[b_offset:b_offset + size] = b_words
    initial_output = 0 if name.startswith("histogram_") else GUARD
    words[out_offset:out_offset + len(expected)] = [initial_output] * len(expected)
    initial = array("I", words).tobytes()
    words[out_offset:out_offset + len(expected)] = expected
    return initial, array("I", words).tobytes(), {
        "name": name, "size": size, "seed": seed,
        "groups": math.ceil(size / THREADS), "threads": THREADS,
        "output_word_offset": out_offset, "output_words": len(expected),
        "output_type": "u32" if name.startswith("histogram_") else "f32",
        "comparison": "bit_exact", "absolute_tolerance": 0, "relative_tolerance": 0,
    }


def repeated_expected(single_expected, case, dispatches):
    if dispatches <= 0:
        raise ValueError("dispatches must be positive")
    if not case["name"].startswith("histogram_"):
        return single_expected
    words = array("I")
    words.frombytes(single_expected)
    start = case["output_word_offset"]
    for index in range(start, start + case["output_words"]):
        value = words[index] * dispatches
        if value > 0xFFFFFFFF:
            raise ValueError("histogram count exceeds u32 across repeated dispatches")
        words[index] = value
    return words.tobytes()


def application_assembly(name):
    lines = [
        f".kernel {name}", ".registers 32", f".workgroup_size {THREADS}, 1, 1",
        "mov_imm r4, 0", "device_load_u32 r3, r4",
        "mov_imm r4, 4", "device_load_u32 r0, r4",
        "mov_imm r4, 8", "device_load_u32 r1, r4",
        "mov_imm r4, 12", "device_load_u32 r2, r4",
        "mov_sr r5, sr_thread_id_x", "mov_sr r6, sr_workgroup_id_x",
        f"mov_imm r7, {THREADS}", "imul r8, r6, r7", "iadd r8, r8, r5",
        "mov_imm r7, 4", "imul r9, r8, r7", "ucmp_lt p1, r8, r3", "if p1",
    ]
    if name == "csr_spmv":
        lines += [
            "mov_imm r4, 16", "device_load_u32 r18, r4",
            "mov_imm r4, 20", "device_load_u32 r19, r4",
            "iadd r10, r0, r9", "device_load_u32 r11, r10",
            "iadd r10, r10, r7", "device_load_u32 r12, r10",
            "mov_imm r13, 0", "mov_imm r20, 1", "loop",
            "icmp_eq p2, r11, r12", "break p2",
            "imul r17, r11, r7", "iadd r10, r1, r17",
            "device_load_u32 r14, r10", "iadd r10, r18, r17",
            "device_load_u32 r15, r10", "imul r14, r14, r7",
            "iadd r10, r19, r14", "device_load_u32 r16, r10",
            "fmul r15, r15, r16", "fadd r13, r13, r15",
            "iadd r11, r11, r20", "endloop",
        ]
    elif name == "stencil_2d":
        lines += [
            "mov_imm r4, 16", "device_load_u32 r20, r4",
            "mov_imm r4, 20", "device_load_u32 r21, r4",
            "iadd r10, r0, r9", "device_load_u32 r13, r10",
            "mov_imm r22, 1", "isub r23, r1, r22", "and r11, r8, r23",
            "shr r12, r8, r21", "mov_imm r22, 0",
            "ucmp_lt p2, r22, r11", "if p2", "ucmp_lt p3, r11, r23", "if p3",
            "ucmp_lt p2, r22, r12", "if p2", "mov_imm r22, 1",
            "isub r23, r20, r22", "ucmp_lt p3, r12, r23", "if p3",
            "imul r14, r1, r7", "isub r15, r10, r14", "device_load_u32 r16, r15",
            "iadd r15, r10, r14", "device_load_u32 r17, r15", "fadd r16, r16, r17",
            "isub r15, r10, r7", "device_load_u32 r17, r15", "fadd r16, r16, r17",
            "iadd r15, r10, r7", "device_load_u32 r17, r15", "fadd r16, r16, r17",
            "mov_imm r18, 1040187392", "fmul r16, r16, r18",
            "mov_imm r18, 1056964608", "fmul r13, r13, r18", "fadd r13, r13, r16",
            "endif", "endif", "endif", "endif",
        ]
    else:
        raise ValueError(f"unknown application workload: {name}")
    lines += ["iadd r10, r2, r9", "device_store_u32 r10, r13", "endif", "halt", ".end", ""]
    return "\n".join(lines)


def application_native_source(name):
    prefix = f"""#include <metal_stdlib>
using namespace metal;
kernel void {name}(device uint* memory [[buffer(0)]], uint i [[thread_position_in_grid]]) {{
    if (i >= memory[0]) return;
    device float* output = (device float*)((device uchar*)memory + memory[3]);
"""
    if name == "csr_spmv":
        body = """    device uint* row_ptr = (device uint*)((device uchar*)memory + memory[1]);
    device uint* col_idx = (device uint*)((device uchar*)memory + memory[2]);
    device float* values = (device float*)((device uchar*)memory + memory[4]);
    device float* x = (device float*)((device uchar*)memory + memory[5]);
    float sum = 0.0f;
    for (uint j = row_ptr[i]; j < row_ptr[i + 1]; ++j) sum += values[j] * x[col_idx[j]];
    output[i] = sum;
"""
    elif name == "stencil_2d":
        body = """    device float* input = (device float*)((device uchar*)memory + memory[1]);
    const uint width = memory[2], height = memory[4];
    const uint x = i & (width - 1), y = i >> memory[5];
    float value = input[i];
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        float neighbors = ((input[i - width] + input[i + width]) + input[i - 1]) + input[i + 1];
        value = value * 0.5f + neighbors * 0.125f;
    }
    output[i] = value;
"""
    else:
        raise ValueError(f"unknown application workload: {name}")
    return prefix + body + "}\n"


def guarded_allocation(regions, output):
    words = [GUARD] * 32
    offsets = {}
    for name, data in regions:
        offsets[name] = len(words)
        words.extend(data)
        words.extend([GUARD] * 16)
    offsets["output"] = len(words)
    words.extend([GUARD] * len(output))
    words.extend([GUARD] * 16)
    return words, offsets


def make_application_case(name, size, seed):
    if size <= 0:
        raise ValueError("application size must be positive")
    if name == "csr_spmv":
        rng = random.Random(seed)
        row_ptr, col_idx, value_numerators = [0], [], []
        lengths = (0, 1, 2, 3, 4, 7, 12, 23)
        vector_numerators = [((i * 17 + 3) % 63) - 31 for i in range(size)]
        expected = []
        for row in range(size):
            length = min(size, lengths[(row + seed) % len(lengths)])
            columns = sorted(rng.sample(range(size), length))
            numerators = [rng.randrange(-7, 8) for _ in columns]
            col_idx.extend(columns)
            value_numerators.extend(numerators)
            row_ptr.append(len(col_idx))
            total = sum(value * vector_numerators[column] for value, column in zip(numerators, columns))
            expected.append(float_word(total / 128))
        values = [float_word(value / 16) for value in value_numerators]
        vector = [float_word(value / 8) for value in vector_numerators]
        words, offsets = guarded_allocation([
            ("row_ptr", row_ptr), ("col_idx", col_idx), ("values", values), ("vector", vector),
        ], expected)
        words[:8] = [size, offsets["row_ptr"] * 4, offsets["col_idx"] * 4,
                     offsets["output"] * 4, offsets["values"] * 4,
                     offsets["vector"] * 4, size, len(col_idx)]
        parameters = {"rows": size, "columns": size, "nonzeros": len(col_idx),
                      "row_length_family": list(lengths), "value_denominator": 16,
                      "vector_denominator": 8, "operation": "y = A x in compressed sparse row format"}
    elif name == "stencil_2d":
        width = 1 << (math.isqrt(size).bit_length() - 1)
        height = (size + width - 1) // width
        numerators = [((i * 37 + 11 + seed % 251) % 251) - 125 for i in range(width * height)]
        expected = []
        for row in range(height):
            for column in range(width):
                index = row * width + column
                if row in (0, height - 1) or column in (0, width - 1):
                    value = numerators[index] / 16
                else:
                    neighbor_sum = sum(numerators[(row + dy) * width + column + dx]
                                       for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))
                    value = (4 * numerators[index] + neighbor_sum) / 128
                expected.append(float_word(value))
        inputs = [float_word(value / 16) for value in numerators]
        words, offsets = guarded_allocation([("input", inputs)], expected)
        words[:6] = [width * height, offsets["input"] * 4, width,
                     offsets["output"] * 4, height, width.bit_length() - 1]
        parameters = {"width": width, "height": height, "cells": width * height,
                      "input_denominator": 16, "boundary": "copy input perimeter",
                      "operation": "center / 2 + (north + south + west + east) / 8"}
    else:
        raise ValueError(f"unknown application workload: {name}")
    initial = array("I", words).tobytes()
    start = offsets["output"]
    words[start:start + len(expected)] = expected
    return initial, array("I", words).tobytes(), {
        "name": name, "size": size, "seed": seed,
        "groups": math.ceil(len(expected) / THREADS), "threads": THREADS,
        "output_word_offset": start, "output_words": len(expected), "output_type": "f32",
        "comparison": "bit_exact", "absolute_tolerance": 0, "relative_tolerance": 0,
        "parameters": parameters, "region_word_offsets": offsets,
    }
