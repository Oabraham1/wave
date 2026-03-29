// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Register naming and special register mapping. Converts WAVE register indices to
//!
//! MSL variable names and maps WAVE special registers to Metal kernel parameter
//! expressions. General registers become uint32_t locals (r0, r1, ...) and predicate
//! registers become bool locals (p0-p3). Special registers map to Metal built-in
//! kernel parameters like thread_position_in_threadgroup and simdgroup_index.

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
        "    uint wave_count = (tsize.x * tsize.y * tsize.z + 31u) / 32u;"
    )
    .unwrap();

    if local_mem_size > 0 {
        writeln!(out, "    threadgroup uint8_t local_mem[{local_mem_size}];").unwrap();
    }

    out
}
