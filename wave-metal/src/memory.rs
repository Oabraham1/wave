// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Memory operation MSL emission. Generates Metal code for WAVE local (threadgroup)
//!
//! and device memory operations including loads, stores, and atomics. Memory accesses
//! use byte offsets from the base pointer with pointer casts to the appropriate width
//! type. Atomics use Metal's `atomic_fetch_*_explicit` functions with `memory_order_relaxed`.

use crate::registers::reg;
use wave_decode::opcodes::{AtomicOp, MemWidth};

#[must_use]
pub fn msl_type(width: MemWidth) -> &'static str {
    match width {
        MemWidth::U8 => "uint8_t",
        MemWidth::U16 => "uint16_t",
        MemWidth::U32 => "uint32_t",
        MemWidth::U64 => "ulong",
        MemWidth::U128 => "uint4",
    }
}

#[must_use]
pub fn emit_load(space: &str, mem_name: &str, width: MemWidth, rd: u8, addr: u8) -> Vec<String> {
    let r_d = reg(rd);
    let r_addr = reg(addr);
    let ty = msl_type(width);

    match width {
        MemWidth::U8 | MemWidth::U16 | MemWidth::U32 => {
            vec![format!(
                "{r_d} = (uint32_t)(*({space} {ty}*)({mem_name} + {r_addr}));"
            )]
        }
        MemWidth::U64 => {
            let r_d1 = reg(rd + 1);
            vec![
                "{".to_string(),
                format!("    ulong tmp_load = *({space} ulong*)({mem_name} + {r_addr});"),
                format!("    {r_d} = (uint32_t)(tmp_load & 0xFFFFFFFFu);"),
                format!("    {r_d1} = (uint32_t)(tmp_load >> 32);"),
                "}".to_string(),
            ]
        }
        MemWidth::U128 => {
            let r_d1 = reg(rd + 1);
            let r_d2 = reg(rd + 2);
            let r_d3 = reg(rd + 3);
            vec![
                "{".to_string(),
                format!("    uint4 tmp_load = *({space} uint4*)({mem_name} + {r_addr});"),
                format!("    {r_d} = tmp_load.x;"),
                format!("    {r_d1} = tmp_load.y;"),
                format!("    {r_d2} = tmp_load.z;"),
                format!("    {r_d3} = tmp_load.w;"),
                "}".to_string(),
            ]
        }
    }
}

#[must_use]
pub fn emit_store(
    space: &str,
    mem_name: &str,
    width: MemWidth,
    addr: u8,
    value: u8,
) -> Vec<String> {
    let r_addr = reg(addr);
    let r_val = reg(value);
    let ty = msl_type(width);

    match width {
        MemWidth::U8 | MemWidth::U16 | MemWidth::U32 => {
            vec![format!(
                "*({space} {ty}*)({mem_name} + {r_addr}) = ({ty}){r_val};"
            )]
        }
        MemWidth::U64 => {
            let r_val1 = reg(value + 1);
            vec![format!(
                "*({space} ulong*)({mem_name} + {r_addr}) = (ulong){r_val} | ((ulong){r_val1} << 32);"
            )]
        }
        MemWidth::U128 => {
            let r_val1 = reg(value + 1);
            let r_val2 = reg(value + 2);
            let r_val3 = reg(value + 3);
            vec![format!(
                "*({space} uint4*)({mem_name} + {r_addr}) = uint4({r_val}, {r_val1}, {r_val2}, {r_val3});"
            )]
        }
    }
}

#[must_use]
pub fn atomic_func(op: AtomicOp) -> &'static str {
    match op {
        AtomicOp::Add => "atomic_fetch_add_explicit",
        AtomicOp::Sub => "atomic_fetch_sub_explicit",
        AtomicOp::Min => "atomic_fetch_min_explicit",
        AtomicOp::Max => "atomic_fetch_max_explicit",
        AtomicOp::And => "atomic_fetch_and_explicit",
        AtomicOp::Or => "atomic_fetch_or_explicit",
        AtomicOp::Xor => "atomic_fetch_xor_explicit",
        AtomicOp::Exchange => "atomic_exchange_explicit",
    }
}

#[must_use]
pub fn emit_atomic(
    space: &str,
    mem_name: &str,
    op: AtomicOp,
    rd: Option<u8>,
    addr: u8,
    value: u8,
) -> Vec<String> {
    let r_addr = reg(addr);
    let r_val = reg(value);
    let func = atomic_func(op);

    match rd {
        Some(d) => {
            let r_d = reg(d);
            vec![format!(
                "{r_d} = {func}(({space} atomic_uint*)({mem_name} + {r_addr}), {r_val}, memory_order_relaxed);"
            )]
        }
        None => {
            vec![format!(
                "{func}(({space} atomic_uint*)({mem_name} + {r_addr}), {r_val}, memory_order_relaxed);"
            )]
        }
    }
}

#[must_use]
pub fn emit_atomic_cas(
    space: &str,
    mem_name: &str,
    rd: Option<u8>,
    addr: u8,
    expected: u8,
    desired: u8,
) -> Vec<String> {
    let r_addr = reg(addr);
    let r_exp = reg(expected);
    let r_des = reg(desired);

    let mut lines = vec![
        "{".to_string(),
        format!("    uint expected_tmp = {r_exp};"),
        format!(
            "    atomic_compare_exchange_weak_explicit(({space} atomic_uint*)({mem_name} + {r_addr}), &expected_tmp, {r_des}, memory_order_relaxed, memory_order_relaxed);"
        ),
    ];

    if let Some(d) = rd {
        lines.push(format!("    {} = expected_tmp;", reg(d)));
    }

    lines.push("}".to_string());
    lines
}
