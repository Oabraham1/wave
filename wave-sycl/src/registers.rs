// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Register naming and special register mapping for SYCL. WAVE general-purpose
//!
//! registers become uint32_t locals and predicates become bool locals. Special
//! registers map to SYCL nd_item accessors (get_local_id, get_group, get_local_range,
//! get_group_range) and sub_group accessors (get_local_id, get_group_id,
//! get_max_local_range, get_group_range) for lane/wave queries.

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
        0 => "it.get_local_id(0)",
        1 => "it.get_local_id(1)",
        2 => "it.get_local_id(2)",
        3 => "sg.get_group_id()[0]",
        4 => "sg.get_local_id()[0]",
        5 => "it.get_group(0)",
        6 => "it.get_group(1)",
        7 => "it.get_group(2)",
        8 => "it.get_local_range(0)",
        9 => "it.get_local_range(1)",
        10 => "it.get_local_range(2)",
        11 => "it.get_group_range(0)",
        12 => "it.get_group_range(1)",
        13 => "it.get_group_range(2)",
        14 => "sg.get_max_local_range()[0]",
        15 => "sg.get_group_range()[0]",
        _ => "0u",
    }
}

#[must_use]
pub fn emit_declarations(reg_count: u32) -> String {
    let mut out = String::new();

    if reg_count > 0 {
        let regs_per_line = 8;
        let mut i = 0u32;
        while i < reg_count {
            let end = (i + regs_per_line).min(reg_count);
            write!(out, "                uint32_t").unwrap();
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
        "                bool p0 = false, p1 = false, p2 = false, p3 = false;"
    )
    .unwrap();

    out
}
