// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Map WAVE registers and thread identity values to Metal.

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
        0 => "tid.x",
        1 => "tid.y",
        2 => "tid.z",
        3 => "simd_id",
        4 => "lane_id",
        5 => "gid.x",
        6 => "gid.y",
        7 => "gid.z",
        8 => "tsize.x",
        9 => "tsize.y",
        10 => "tsize.z",
        11 => "grid_dim.x",
        12 => "grid_dim.y",
        13 => "grid_dim.z",
        14 => "32u",
        15 => "wave_count",
        _ => "0u",
    }
}

#[must_use]
pub fn emit_declarations(reg_count: u32, local_mem_size: u32) -> String {
    let mut out = String::new();

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
        "    uint wave_count = (tsize.x * tsize.y * tsize.z + 31u) / 32u;"
    )
    .expect("writing to String cannot fail");

    if local_mem_size > 0 {
        writeln!(out, "    threadgroup uint8_t local_mem[{local_mem_size}];")
            .expect("writing to String cannot fail");
        writeln!(out, "    float _mma_a[16] = {{}};").expect("writing to String cannot fail");
        writeln!(out, "    float _mma_b[16] = {{}};").expect("writing to String cannot fail");
        writeln!(out, "    float _mma_c[16] = {{}};").expect("writing to String cannot fail");
    }

    out
}
