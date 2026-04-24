// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Register naming and special register mapping for HIP. WAVE general-purpose registers
//!
//! become `uint32_t` locals (r0, r1, ...) and predicates become bool locals (p0-p3).
//! Special registers map to HIP built-in variables: threadIdx for thread IDs, blockIdx
//! for workgroup IDs, blockDim for workgroup sizes, gridDim for grid sizes, `__lane_id()`
//! for lane ID, and warpSize for wavefront width (32 on RDNA, 64 on CDNA).

use std::fmt::Write;

#[must_use]
pub fn reg(index: u8) -> String {
    format!("r{index}")
}

#[must_use]
pub fn pred(index: u8) -> String {
    format!("p{index}")
}

#[must_use]
pub fn special_reg_expr(sr_index: u8) -> &'static str {
    match sr_index {
        0 => "threadIdx.x",
        1 => "threadIdx.y",
        2 => "threadIdx.z",
        3 => "wave_id",
        4 => "__lane_id()",
        5 => "blockIdx.x",
        6 => "blockIdx.y",
        7 => "blockIdx.z",
        8 => "blockDim.x",
        9 => "blockDim.y",
        10 => "blockDim.z",
        11 => "gridDim.x",
        12 => "gridDim.y",
        13 => "gridDim.z",
        14 => "warpSize",
        15 => "wave_count",
        _ => "0u",
    }
}

#[must_use]
pub fn emit_declarations(reg_count: u32, has_local_mem: bool) -> String {
    let mut out = String::new();

    if has_local_mem {
        writeln!(out, "    extern __shared__ uint8_t local_mem[];").unwrap();
        writeln!(out, "    float _mma_a[16] = {{}};").unwrap();
        writeln!(out, "    float _mma_b[16] = {{}};").unwrap();
        writeln!(out, "    float _mma_c[16] = {{}};").unwrap();
    }

    if reg_count > 0 {
        let regs_per_line = 8;
        let mut i = 0u32;
        while i < reg_count {
            let end = (i + regs_per_line).min(reg_count);
            write!(out, "    uint32_t").unwrap();
            for j in i..end {
                if j > i {
                    write!(out, ",").unwrap();
                }
                write!(out, " r{j} = 0").unwrap();
            }
            writeln!(out, ";").unwrap();
            i = end;
        }
    }

    writeln!(
        out,
        "    bool p0 = false, p1 = false, p2 = false, p3 = false;"
    )
    .unwrap();
    writeln!(
        out,
        "    uint32_t wave_id = (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) / warpSize;"
    )
    .unwrap();
    writeln!(
        out,
        "    uint32_t wave_count = (blockDim.x * blockDim.y * blockDim.z + warpSize - 1) / warpSize;"
    )
    .unwrap();

    out
}
