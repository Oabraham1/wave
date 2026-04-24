#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Parameterized WAVE GEMM kernel generator for auto-tuning.
#
# Generates .wave assembly files for blocked GEMM with configurable tile sizes,
# register blocking factors, workgroup dimensions, and optional double-buffered
# prefetching. Supports five kernel variants: F32, F16, bias+ReLU, bias+GELU,
# and F16+bias+ReLU. The tuning parameters control the GEMM core; epilogues
# are appended unchanged.

import argparse
import json
import math
import sys


def log2_int(n):
    assert n > 0 and (n & (n - 1)) == 0, f"{n} is not a power of 2"
    return int(math.log2(n))


KERNEL_NAMES = {
    "f32": "gemm_register_blocked_8x8",
    "f16": "gemm_register_blocked_8x8_f16",
    "bias_relu": "gemm_bias_relu_fused",
    "bias_gelu": "gemm_bias_gelu_fused",
    "bias_relu_f16": "gemm_bias_relu_f16_fused",
}

ACC_START = 16


def validate_config(tile_m, tile_n, tile_k, block_m, block_n, prefetch):
    wg_x = tile_n // block_n
    wg_y = tile_m // block_m
    threads = wg_x * wg_y
    if threads > 1024:
        return False, "workgroup exceeds 1024"
    num_regs = ACC_START + block_m * block_n + block_n + 8
    if num_regs > 255:
        return False, "register overflow"
    if (tile_m * tile_k) % threads != 0:
        return False, "A loads not divisible"
    if (tile_k * tile_n) % threads != 0:
        return False, "B loads not divisible"
    a_loads = tile_m * tile_k // threads
    b_loads = tile_k * tile_n // threads
    if a_loads < 1 or b_loads < 1:
        return False, "loads < 1"
    a_buf = tile_m * tile_k * 4
    b_buf = tile_k * tile_n * 4
    if prefetch:
        local_mem = 2 * (a_buf + b_buf) + 16
    else:
        local_mem = a_buf + b_buf + 16
    if local_mem > 32768:
        return False, "local memory exceeds 32KB"
    return True, "ok"


def enumerate_configs():
    tile_pairs = [
        (64, 64), (128, 128), (256, 256),
        (64, 128), (128, 64), (128, 256), (256, 128),
    ]
    tile_ks = [4, 8, 16]
    block_pairs = [(4, 4), (8, 8), (4, 8), (8, 4)]
    prefetches = [True, False]

    configs = []
    for tm, tn in tile_pairs:
        for tk in tile_ks:
            for bm, bn in block_pairs:
                for pf in prefetches:
                    ok, _ = validate_config(tm, tn, tk, bm, bn, pf)
                    if ok:
                        configs.append({
                            "tile_m": tm, "tile_n": tn, "tile_k": tk,
                            "block_m": bm, "block_n": bn, "prefetch": pf,
                        })
    return configs


def _emit_tile_load_a(L, tile_m, tile_k, threads, a_loads, elem, ld,
                      T, param_base, k_source, buf_dest_reg):
    a_rows_per_batch = threads // tile_k
    log2_tk = log2_int(tile_k)
    addr = T if k_source == "zero" else T + 6

    L(f"    mov_imm r{T}, {param_base}")
    L(f"    local_load_u32 r11, r{T}")
    L("")
    L(f"    mov_imm r{T}, {tile_k - 1}")
    L(f"    and r12, r9, r{T}")
    L(f"    mov_imm r{T}, {log2_tk}")
    L(f"    shr r13, r9, r{T}")
    L("")
    L(f"    mov_sr r{T}, sr_workgroup_id_y")
    L(f"    mov_imm r{T + 1}, {tile_m}")
    L(f"    imul r{T}, r{T}, r{T + 1}")
    L(f"    iadd r14, r{T}, r13")
    L("")

    if k_source == "zero":
        L(f"    imul r{addr}, r14, r6")
        L(f"    iadd r{addr}, r{addr}, r12")
    elif k_source == "r10":
        L(f"    imul r{addr}, r14, r6")
        L(f"    iadd r{addr}, r{addr}, r10")
        L(f"    iadd r{addr}, r{addr}, r12")
    elif k_source == "r10_plus_tk":
        L(f"    mov_imm r{T}, {tile_k}")
        L(f"    iadd r{T}, r10, r{T}")
        L(f"    imul r{addr}, r14, r6")
        L(f"    iadd r{addr}, r{addr}, r{T}")
        L(f"    iadd r{addr}, r{addr}, r12")

    L(f"    mov_imm r{T + 1}, {elem}")
    L(f"    imul r{addr}, r{addr}, r{T + 1}")
    L(f"    iadd r{addr}, r{addr}, r11")
    L("")
    L(f"    mov_imm r{T + 1}, {a_rows_per_batch}")
    L(f"    imul r{T + 1}, r6, r{T + 1}")
    L(f"    mov_imm r{T + 2}, {elem}")
    L(f"    imul r{T + 1}, r{T + 1}, r{T + 2}")
    L("")
    L(f"    mov_imm r{T + 2}, {elem}")
    L(f"    imul r14, r9, r{T + 2}")
    if buf_dest_reg:
        L(f"    iadd r14, r14, r{buf_dest_reg}")
    L("")
    L(f"    mov_imm r{T + 2}, {threads * elem}")
    L("")

    L(f"    device_load_{ld} r{T + 3}, r{addr}")
    L(f"    local_store_{ld} r14, r{T + 3}")
    for i in range(1, a_loads):
        L("")
        L(f"    iadd r{addr}, r{addr}, r{T + 1}")
        L(f"    iadd r14, r14, r{T + 2}")
        L(f"    device_load_{ld} r{T + 3}, r{addr}")
        L(f"    local_store_{ld} r14, r{T + 3}")


def _emit_tile_load_b(L, tile_n, tile_k, threads, b_loads, elem, ld,
                      T, param_base, param_n_off, a_buf, k_source, buf_dest_reg):
    b_rows_per_batch = threads // tile_n
    log2_tn = log2_int(tile_n)

    L(f"    mov_imm r{T}, {param_base + 4}")
    L(f"    local_load_u32 r11, r{T}")
    L(f"    mov_imm r{T}, {param_n_off}")
    L(f"    local_load_u32 r12, r{T}")
    L("")
    L(f"    mov_imm r{T}, {tile_n - 1}")
    L(f"    and r13, r9, r{T}")
    L(f"    mov_imm r{T}, {log2_tn}")
    L(f"    shr r14, r9, r{T}")
    L("")

    if k_source == "zero":
        L(f"    mov r{T}, r14")
    elif k_source == "r10":
        L(f"    iadd r{T}, r10, r14")
    elif k_source == "r10_plus_tk":
        L(f"    mov_imm r{T}, {tile_k}")
        L(f"    iadd r{T}, r10, r{T}")
        L(f"    iadd r{T}, r{T}, r14")

    L(f"    mov_sr r{T + 1}, sr_workgroup_id_x")
    L(f"    mov_imm r{T + 2}, {tile_n}")
    L(f"    imul r{T + 1}, r{T + 1}, r{T + 2}")
    L(f"    iadd r{T + 1}, r{T + 1}, r13")
    L(f"    imul r{T}, r{T}, r12")
    L(f"    iadd r{T}, r{T}, r{T + 1}")
    L("")
    L(f"    mov_imm r{T + 2}, {elem}")
    L(f"    imul r{T}, r{T}, r{T + 2}")
    L(f"    iadd r{T}, r{T}, r11")
    L("")
    L(f"    mov_imm r{T + 1}, {b_rows_per_batch}")
    L(f"    imul r{T + 1}, r12, r{T + 1}")
    L(f"    mov_imm r{T + 2}, {elem}")
    L(f"    imul r{T + 1}, r{T + 1}, r{T + 2}")
    L("")
    L(f"    mov_imm r{T + 2}, {elem}")
    L(f"    imul r14, r9, r{T + 2}")
    L(f"    mov_imm r{T + 2}, {a_buf}")
    L(f"    iadd r14, r14, r{T + 2}")
    if buf_dest_reg:
        L(f"    iadd r14, r14, r{buf_dest_reg}")
    L(f"    mov_imm r{T + 2}, {threads * elem}")
    L("")

    L(f"    device_load_{ld} r{T + 3}, r{T}")
    L(f"    local_store_{ld} r14, r{T + 3}")
    for i in range(1, b_loads):
        L("")
        L(f"    iadd r{T}, r{T}, r{T + 1}")
        L(f"    iadd r14, r14, r{T + 2}")
        L(f"    device_load_{ld} r{T + 3}, r{T}")
        L(f"    local_store_{ld} r14, r{T + 3}")


def _emit_inner_compute(L, tile_m, tile_n, tile_k, block_m, block_n,
                        elem, ld, is_f16, a_buf, T, B_S, A_R):
    a_row_stride = tile_k * elem

    L(f"    mov_imm r{T}, {block_m * tile_k * elem}")
    L(f"    imul r12, r8, r{T}")
    L(f"    iadd r12, r12, r15")
    L("")
    L(f"    mov_imm r{T}, {block_n * elem}")
    L(f"    imul r13, r7, r{T}")
    L(f"    mov_imm r{T}, {a_buf}")
    L(f"    iadd r13, r13, r{T}")
    L(f"    iadd r13, r13, r15")
    L("")
    L(f"    mov_imm r14, {tile_k * elem}")
    L(f"    iadd r14, r12, r14")
    L("")
    L(f"    loop")
    L(f"        icmp_ge p0, r12, r14")
    L(f"        break p0")
    L("")

    L(f"        mov_imm r11, {elem}")
    L(f"        mov r{T}, r13")
    for j in range(block_n):
        L(f"        local_load_{ld} r{B_S + j}, r{T}")
        if is_f16:
            L(f"        cvt_f32_f16 r{B_S + j}, r{B_S + j}")
        if j < block_n - 1:
            L(f"        iadd r{T}, r{T}, r11")
    L("")

    for i in range(block_m):
        if i == 0:
            L(f"        local_load_{ld} r{A_R}, r12")
        else:
            L(f"        mov_imm r{T}, {i * a_row_stride}")
            L(f"        iadd r{T}, r12, r{T}")
            L(f"        local_load_{ld} r{A_R}, r{T}")
        if is_f16:
            L(f"        cvt_f32_f16 r{A_R}, r{A_R}")
        for j in range(block_n):
            acc = ACC_START + i * block_n + j
            L(f"        fma r{acc}, r{A_R}, r{B_S + j}, r{acc}")
        L("")

    L(f"        mov_imm r11, {elem}")
    L(f"        iadd r12, r12, r11")
    L(f"        mov_imm r11, {tile_n * elem}")
    L(f"        iadd r13, r13, r11")
    L(f"    endloop")


def _emit_store(L, tile_m, tile_n, block_m, block_n, elem, ld, is_f16,
                T, param_base, has_bias):
    if has_bias:
        n_off = param_base + 8
        L(f"    mov_imm r{T}, {n_off}")
        L(f"    local_load_u32 r12, r{T}")
    else:
        c_off = param_base + 8
        n_off = param_base + 12
        L(f"    mov_imm r{T}, {c_off}")
        L(f"    local_load_u32 r11, r{T}")
        L(f"    mov_imm r{T}, {n_off}")
        L(f"    local_load_u32 r12, r{T}")
    L("")

    L(f"    mov_sr r{T}, sr_workgroup_id_y")
    L(f"    mov_imm r{T + 1}, {tile_m}")
    L(f"    imul r{T}, r{T}, r{T + 1}")
    L(f"    mov_imm r{T + 1}, {block_m}")
    L(f"    imul r{T + 1}, r8, r{T + 1}")
    L(f"    iadd r13, r{T}, r{T + 1}")
    L("")
    L(f"    mov_sr r{T}, sr_workgroup_id_x")
    L(f"    mov_imm r{T + 1}, {tile_n}")
    L(f"    imul r{T}, r{T}, r{T + 1}")
    L(f"    mov_imm r{T + 1}, {block_n}")
    L(f"    imul r{T + 1}, r7, r{T + 1}")
    L(f"    iadd r14, r{T}, r{T + 1}")
    L("")
    L(f"    mov_imm r{T + 1}, {elem}")
    L(f"    imul r15, r12, r{T + 1}")
    L("")
    L(f"    imul r{T}, r13, r12")
    L(f"    iadd r{T}, r{T}, r14")
    L(f"    mov_imm r{T + 1}, {elem}")
    L(f"    imul r{T}, r{T}, r{T + 1}")

    if has_bias:
        L(f"    iadd r{T}, r{T}, r3")
    else:
        L(f"    iadd r{T}, r{T}, r11")
    L("")
    L(f"    mov_imm r{T + 1}, {elem}")
    L("")

    for i in range(block_m):
        if i > 0:
            L(f"    iadd r{T}, r{T}, r15")
        for j in range(block_n):
            acc = ACC_START + i * block_n + j
            if is_f16:
                L(f"    cvt_f16_f32 r{acc}, r{acc}")
            if j == 0:
                L(f"    device_store_{ld} r{T}, r{acc}")
            elif j == 1:
                L(f"    iadd r{T + 2}, r{T}, r{T + 1}")
                L(f"    device_store_{ld} r{T + 2}, r{acc}")
            else:
                L(f"    iadd r{T + 2}, r{T + 2}, r{T + 1}")
                L(f"    device_store_{ld} r{T + 2}, r{acc}")
        L("")


def _emit_epilogue_bias(L, tile_n, block_m, block_n, T, param_base):
    L(f"    mov_imm r{T}, {param_base + 12}")
    L(f"    local_load_u32 r11, r{T}")
    L(f"    mov_imm r{T}, {param_base + 8}")
    L(f"    local_load_u32 r12, r{T}")
    L("")
    L(f"    mov_sr r{T}, sr_workgroup_id_x")
    L(f"    mov_imm r{T + 1}, {tile_n}")
    L(f"    imul r{T}, r{T}, r{T + 1}")
    L(f"    mov_imm r{T + 1}, {block_n}")
    L(f"    imul r{T + 1}, r7, r{T + 1}")
    L(f"    iadd r{T}, r{T}, r{T + 1}")
    L("")
    L(f"    mov_imm r{T + 1}, 4")
    L(f"    imul r{T}, r{T}, r{T + 1}")
    L(f"    iadd r{T}, r{T}, r11")
    L("")

    B_S = ACC_START + block_m * block_n
    L(f"    mov_imm r{T + 1}, 4")
    L(f"    device_load_u32 r{B_S}, r{T}")
    for j in range(1, block_n):
        L(f"    iadd r{T}, r{T}, r{T + 1}")
        L(f"    device_load_u32 r{B_S + j}, r{T}")
    L("")

    for i in range(block_m):
        for j in range(block_n):
            acc = ACC_START + i * block_n + j
            L(f"    fadd r{acc}, r{acc}, r{B_S + j}")
    L("")


def _emit_epilogue_relu(L, block_m, block_n):
    A_R = ACC_START + block_m * block_n + block_n
    L(f"    mov_imm r{A_R}, 0")
    for i in range(block_m):
        for j in range(block_n):
            acc = ACC_START + i * block_n + j
            L(f"    fmax r{acc}, r{acc}, r{A_R}")
    L("")


def _emit_epilogue_gelu(L, block_m, block_n, T):
    A_R = ACC_START + block_m * block_n + block_n
    L(f"    mov_imm r{T + 1}, 0x401D1463")
    L(f"    mov_imm r{T + 2}, 0x3F800000")
    L("")
    for i in range(block_m):
        for j in range(block_n):
            acc = ACC_START + i * block_n + j
            L(f"    mov r{A_R}, r{acc}")
            L(f"    fmul r{T}, r{A_R}, r{T + 1}")
            L(f"    fneg r{T}, r{T}")
            L(f"    fexp2 r{T}, r{T}")
            L(f"    fadd r{T}, r{T + 2}, r{T}")
            L(f"    frcp r{T}, r{T}")
            L(f"    fmul r{acc}, r{A_R}, r{T}")
            L("")


def generate_gemm(tile_m, tile_n, tile_k, block_m, block_n,
                  prefetch, variant="f32"):
    is_f16 = variant in ("f16", "bias_relu_f16")
    has_bias = variant in ("bias_relu", "bias_gelu", "bias_relu_f16")
    has_relu = variant in ("bias_relu", "bias_relu_f16")
    has_gelu = variant == "bias_gelu"
    elem = 2 if is_f16 else 4
    ld = "u16" if is_f16 else "u32"

    wg_x = tile_n // block_n
    wg_y = tile_m // block_m
    threads = wg_x * wg_y

    a_buf = tile_m * tile_k * elem
    b_buf = tile_k * tile_n * elem
    if prefetch:
        half_buf = a_buf + b_buf
        param_base = 2 * half_buf
    else:
        half_buf = 0
        param_base = a_buf + b_buf
    local_mem = param_base + 16

    a_loads = tile_m * tile_k // threads
    b_loads = tile_k * tile_n // threads

    B_S = ACC_START + block_m * block_n
    A_R = B_S + block_n
    T = A_R + 1
    num_regs = T + 7

    if has_bias:
        param_n_off = param_base + 8
    else:
        param_n_off = param_base + 12

    kname = KERNEL_NAMES[variant]
    lines = []
    L = lines.append

    pf_tag = "double-buffered" if prefetch else "single-buffered"
    L(f"; SPDX-License-Identifier: Apache-2.0")
    L(f";")
    L(f"; Auto-generated parameterized GEMM kernel ({pf_tag}).")
    L(f"; tile={tile_m}x{tile_n}, k={tile_k}, block={block_m}x{block_n}, "
      f"wg={wg_x}x{wg_y}, variant={variant}")
    L("")
    L(f".kernel {kname}")
    L(f".registers {num_regs}")
    L(f".workgroup_size {wg_x}, {wg_y}, 1")
    L(f".local_memory {local_mem}")
    L("")

    L(f"    mov r6, r5")
    L("")
    L(f"    mov_imm r11, {param_base}")
    L(f"    local_store_u32 r11, r0")
    L(f"    mov_imm r11, {param_base + 4}")
    L(f"    local_store_u32 r11, r1")
    if has_bias:
        L(f"    mov_imm r11, {param_base + 8}")
        L(f"    local_store_u32 r11, r4")
        L(f"    mov_imm r11, {param_base + 12}")
        L(f"    local_store_u32 r11, r2")
    else:
        L(f"    mov_imm r11, {param_base + 8}")
        L(f"    local_store_u32 r11, r2")
        L(f"    mov_imm r11, {param_base + 12}")
        L(f"    local_store_u32 r11, r4")
    L("")

    L(f"    mov_sr r7, sr_thread_id_x")
    L(f"    mov_sr r8, sr_thread_id_y")
    L(f"    mov_imm r11, {wg_x}")
    L(f"    imul r9, r8, r11")
    L(f"    iadd r9, r9, r7")
    L("")

    for i in range(block_m * block_n):
        L(f"    mov_imm r{ACC_START + i}, 0")
    L("")

    L(f"    mov_imm r15, 0")
    L("")

    if prefetch:
        _emit_tile_load_a(L, tile_m, tile_k, threads, a_loads, elem, ld,
                          T, param_base, "zero", None)
        L("")
        _emit_tile_load_b(L, tile_n, tile_k, threads, b_loads, elem, ld,
                          T, param_base, param_n_off, a_buf, "zero", None)
        L("")
        L(f"    barrier")
        L("")
        L(f"    mov_imm r10, 0")
        L("")
        L(f"    loop")
        L(f"        icmp_ge p0, r10, r6")
        L(f"        break p0")
        L("")
        L(f"        mov_imm r{T}, {tile_k}")
        L(f"        iadd r{T}, r10, r{T}")
        L(f"        icmp_lt p0, r{T}, r6")
        L(f"        if p0")
        L("")
        L(f"            mov_imm r{T + 4}, {half_buf}")
        L(f"            xor r{T + 5}, r15, r{T + 4}")
        L("")

        indent_save = []
        def L_indent(s):
            indent_save.append(s.replace("    ", "            ", 1) if s.startswith("    ") else s)

        _emit_tile_load_a(L_indent, tile_m, tile_k, threads, a_loads, elem, ld,
                          T, param_base, "r10_plus_tk", T + 5)
        lines.extend(indent_save)
        L("")

        indent_save2 = []
        def L_indent2(s):
            indent_save2.append(s.replace("    ", "            ", 1) if s.startswith("    ") else s)

        _emit_tile_load_b(L_indent2, tile_n, tile_k, threads, b_loads, elem, ld,
                          T, param_base, param_n_off, a_buf, "r10_plus_tk", T + 5)
        lines.extend(indent_save2)
        L("")
        L(f"        endif")
        L("")

        _emit_inner_compute(L, tile_m, tile_n, tile_k, block_m, block_n,
                            elem, ld, is_f16, a_buf, T, B_S, A_R)
        L("")
        L(f"        barrier")
        L("")
        L(f"        mov_imm r{T}, {half_buf}")
        L(f"        xor r15, r15, r{T}")
        L("")
        L(f"        mov_imm r11, {tile_k}")
        L(f"        iadd r10, r10, r11")
        L(f"    endloop")
    else:
        L(f"    mov_imm r10, 0")
        L("")
        L(f"    loop")
        L(f"        icmp_ge p0, r10, r6")
        L(f"        break p0")
        L("")

        indent_np = []
        def L_np(s):
            indent_np.append(s.replace("    ", "        ", 1) if s.startswith("    ") else s)

        _emit_tile_load_a(L_np, tile_m, tile_k, threads, a_loads, elem, ld,
                          T, param_base, "r10", None)
        lines.extend(indent_np)
        L("")

        indent_np2 = []
        def L_np2(s):
            indent_np2.append(s.replace("    ", "        ", 1) if s.startswith("    ") else s)

        _emit_tile_load_b(L_np2, tile_n, tile_k, threads, b_loads, elem, ld,
                          T, param_base, param_n_off, a_buf, "r10", None)
        lines.extend(indent_np2)
        L("")
        L(f"        barrier")
        L("")

        _emit_inner_compute(L, tile_m, tile_n, tile_k, block_m, block_n,
                            elem, ld, is_f16, a_buf, T, B_S, A_R)
        L("")
        L(f"        barrier")
        L("")
        L(f"        mov_imm r11, {tile_k}")
        L(f"        iadd r10, r10, r11")
        L(f"    endloop")

    L("")

    if has_bias:
        _emit_epilogue_bias(L, tile_n, block_m, block_n, T, param_base)
        if has_relu:
            _emit_epilogue_relu(L, block_m, block_n)
        elif has_gelu:
            _emit_epilogue_gelu(L, block_m, block_n, T)

    _emit_store(L, tile_m, tile_n, block_m, block_n, elem, ld, is_f16,
                T, param_base, has_bias)

    L(f"    halt")
    L(f".end")
    L("")
    return "\n".join(lines)


def config_tag(cfg):
    p = "p" if cfg["prefetch"] else "np"
    return (f"t{cfg['tile_m']}x{cfg['tile_n']}_k{cfg['tile_k']}"
            f"_b{cfg['block_m']}x{cfg['block_n']}_{p}")


def main():
    parser = argparse.ArgumentParser(description="Generate parameterized WAVE GEMM kernels")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate a single kernel")
    gen.add_argument("--tile-m", type=int, required=True)
    gen.add_argument("--tile-n", type=int, required=True)
    gen.add_argument("--tile-k", type=int, required=True)
    gen.add_argument("--block-m", type=int, required=True)
    gen.add_argument("--block-n", type=int, required=True)
    gen.add_argument("--prefetch", action="store_true", default=False)
    gen.add_argument("--variant", default="f32",
                     choices=["f32", "f16", "bias_relu", "bias_gelu", "bias_relu_f16"])
    gen.add_argument("-o", "--output", default="-")

    lst = sub.add_parser("list", help="List all valid configurations")
    lst.add_argument("--json", action="store_true", default=False)

    args = parser.parse_args()

    if args.command == "generate":
        ok, reason = validate_config(args.tile_m, args.tile_n, args.tile_k,
                                     args.block_m, args.block_n, args.prefetch)
        if not ok:
            print(f"Invalid config: {reason}", file=sys.stderr)
            sys.exit(1)
        asm = generate_gemm(args.tile_m, args.tile_n, args.tile_k,
                            args.block_m, args.block_n, args.prefetch,
                            args.variant)
        if args.output == "-":
            print(asm, end="")
        else:
            with open(args.output, "w") as f:
                f.write(asm)

    elif args.command == "list":
        configs = enumerate_configs()
        if args.json:
            print(json.dumps(configs, indent=2))
        else:
            print(f"Valid configurations: {len(configs)}")
            for c in configs:
                print(f"  {config_tag(c)}  wg={c['tile_n']//c['block_n']}x{c['tile_m']//c['block_m']}"
                      f"  regs={ACC_START + c['block_m']*c['block_n'] + c['block_n'] + 8}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
