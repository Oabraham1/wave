// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Map WAVE registers and thread identity values to HIP C++.

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
        writeln!(out, "    extern __shared__ uint8_t local_mem[];")
            .expect("writing to String cannot fail");
        writeln!(out, "    float _mma_a[16] = {{}};").expect("writing to String cannot fail");
        writeln!(out, "    float _mma_b[16] = {{}};").expect("writing to String cannot fail");
        writeln!(out, "    float _mma_c[16] = {{}};").expect("writing to String cannot fail");
    }

    if reg_count > 0 {
        let regs_per_line = 8;
        let mut i = 0u32;
        while i < reg_count {
            let end = (i + regs_per_line).min(reg_count);
            write!(out, "    uint32_t").expect("writing to String cannot fail");
            for j in i..end {
                if j > i {
                    write!(out, ",").expect("writing to String cannot fail");
                }
                write!(out, " r{j} = 0").expect("writing to String cannot fail");
            }
            writeln!(out, ";").expect("writing to String cannot fail");
            i = end;
        }
    }

    writeln!(
        out,
        "    bool p0 = false, p1 = false, p2 = false, p3 = false;"
    )
    .expect("writing to String cannot fail");
    writeln!(
        out,
        "    uint32_t wave_id = (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) / warpSize;"
    )
    .expect("writing to String cannot fail");
    writeln!(
        out,
        "    uint32_t wave_count = (blockDim.x * blockDim.y * blockDim.z + warpSize - 1) / warpSize;"
    )
    .expect("writing to String cannot fail");

    out
}
