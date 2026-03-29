// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! PTX register naming and special register mapping. WAVE general-purpose registers
//!
//! map to %r (b32) and %f (f32 shadow) PTX registers. Predicates map to %p (pred).
//! Address computation uses %rd (b64) registers. Special registers map to PTX built-in
//! registers like %tid.x, %ctaid.x, %laneid. Computed specials (wave_id, num_waves)
//! emit multi-instruction sequences using temporary registers.

use std::fmt::Write;

#[must_use]
pub fn reg(index: u8) -> String {
    format!("%r{index}")
}

#[must_use]
pub fn freg(index: u8) -> String {
    format!("%f{index}")
}

#[must_use]
pub fn pred(index: u8) -> String {
    format!("%p{index}")
}

#[must_use]
pub fn dreg(index: u8) -> String {
    format!("%rd{index}")
}

#[must_use]
pub fn emit_special_reg(rd: u8, sr_index: u8) -> Vec<String> {
    let r_d = reg(rd);
    match sr_index {
        0 => vec![format!("mov.u32 {r_d}, %tid.x;")],
        1 => vec![format!("mov.u32 {r_d}, %tid.y;")],
        2 => vec![format!("mov.u32 {r_d}, %tid.z;")],
        3 => vec![
            "mov.u32 %t0, %tid.x;".to_string(),
            "mov.u32 %t1, %tid.y;".to_string(),
            "mov.u32 %t2, %ntid.x;".to_string(),
            "mad.lo.u32 %t0, %t1, %t2, %t0;".to_string(),
            "mov.u32 %t1, %tid.z;".to_string(),
            "mov.u32 %t3, %ntid.y;".to_string(),
            "mul.lo.u32 %t2, %t2, %t3;".to_string(),
            "mad.lo.u32 %t0, %t1, %t2, %t0;".to_string(),
            format!("shr.u32 {r_d}, %t0, 5;"),
        ],
        4 => vec![format!("mov.u32 {r_d}, %laneid;")],
        5 => vec![format!("mov.u32 {r_d}, %ctaid.x;")],
        6 => vec![format!("mov.u32 {r_d}, %ctaid.y;")],
        7 => vec![format!("mov.u32 {r_d}, %ctaid.z;")],
        8 => vec![format!("mov.u32 {r_d}, %ntid.x;")],
        9 => vec![format!("mov.u32 {r_d}, %ntid.y;")],
        10 => vec![format!("mov.u32 {r_d}, %ntid.z;")],
        11 => vec![format!("mov.u32 {r_d}, %nctaid.x;")],
        12 => vec![format!("mov.u32 {r_d}, %nctaid.y;")],
        13 => vec![format!("mov.u32 {r_d}, %nctaid.z;")],
        14 => vec![format!("mov.u32 {r_d}, 32;")],
        15 => vec![
            "mov.u32 %t0, %ntid.x;".to_string(),
            "mov.u32 %t1, %ntid.y;".to_string(),
            "mul.lo.u32 %t0, %t0, %t1;".to_string(),
            "mov.u32 %t1, %ntid.z;".to_string(),
            "mul.lo.u32 %t0, %t0, %t1;".to_string(),
            "add.u32 %t0, %t0, 31;".to_string(),
            format!("shr.u32 {r_d}, %t0, 5;"),
        ],
        _ => vec![format!("mov.u32 {r_d}, 0;")],
    }
}

#[must_use]
pub fn emit_declarations(reg_count: u32, local_mem_size: u32) -> String {
    let mut out = String::new();

    writeln!(out, "    .reg .b32 %r<{reg_count}>;").unwrap();
    writeln!(out, "    .reg .f32 %f<{reg_count}>;").unwrap();
    writeln!(out, "    .reg .pred %p<4>;").unwrap();
    writeln!(out, "    .reg .b64 %rd<4>;").unwrap();
    writeln!(out, "    .reg .b32 %t<8>;").unwrap();
    writeln!(out, "    .reg .f32 %ft<4>;").unwrap();
    writeln!(out, "    .reg .pred %pt<2>;").unwrap();

    if local_mem_size > 0 {
        writeln!(
            out,
            "    .shared .align 4 .b8 _shared_mem[{local_mem_size}];"
        )
        .unwrap();
    }

    writeln!(out).unwrap();
    writeln!(out, "    ld.param.u64 %rd0, [_device_mem_ptr];").unwrap();

    out
}
