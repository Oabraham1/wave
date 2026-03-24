// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! PTX memory operation emission. Generates load/store instructions for global (device)
//!
//! and shared (local/threadgroup) memory spaces. Global accesses compute 64-bit
//! addresses by zero-extending the 32-bit WAVE byte offset and adding it to the base
//! pointer in %rd0. Shared accesses offset from the _shared_mem symbol. Atomics map
//! to PTX atom.global/atom.shared instructions; atomic_sub is implemented via negation
//! followed by atom.add since PTX has no atom.sub.

use crate::registers::reg;
use wave_decode::opcodes::{AtomicOp, MemWidth};

#[must_use]
pub fn ptx_width(w: MemWidth) -> &'static str {
    match w {
        MemWidth::U8 => "u8",
        MemWidth::U16 => "u16",
        MemWidth::U32 => "u32",
        MemWidth::U64 => "u64",
        MemWidth::U128 => "b128",
    }
}

#[must_use]
pub fn emit_global_load(w: MemWidth, rd: u8, addr: u8) -> Vec<String> {
    let r_d = reg(rd);
    let r_addr = reg(addr);
    let pw = ptx_width(w);

    match w {
        MemWidth::U8 | MemWidth::U16 | MemWidth::U32 => vec![
            format!("cvt.u64.u32 %rd1, {r_addr};"),
            "add.u64 %rd1, %rd0, %rd1;".to_string(),
            format!("ld.global.{pw} {r_d}, [%rd1];"),
        ],
        MemWidth::U64 => {
            let r_d1 = reg(rd + 1);
            vec![
                format!("cvt.u64.u32 %rd1, {r_addr};"),
                "add.u64 %rd1, %rd0, %rd1;".to_string(),
                "ld.global.u64 %rd2, [%rd1];".to_string(),
                format!("mov.b64 {{{r_d}, {r_d1}}}, %rd2;"),
            ]
        }
        MemWidth::U128 => {
            let r_d1 = reg(rd + 1);
            let r_d2 = reg(rd + 2);
            let r_d3 = reg(rd + 3);
            vec![
                format!("cvt.u64.u32 %rd1, {r_addr};"),
                "add.u64 %rd1, %rd0, %rd1;".to_string(),
                format!("ld.global.v4.u32 {{{r_d}, {r_d1}, {r_d2}, {r_d3}}}, [%rd1];"),
            ]
        }
    }
}

#[must_use]
pub fn emit_global_store(w: MemWidth, addr: u8, value: u8) -> Vec<String> {
    let r_addr = reg(addr);
    let r_val = reg(value);
    let pw = ptx_width(w);

    match w {
        MemWidth::U8 | MemWidth::U16 | MemWidth::U32 => vec![
            format!("cvt.u64.u32 %rd1, {r_addr};"),
            "add.u64 %rd1, %rd0, %rd1;".to_string(),
            format!("st.global.{pw} [%rd1], {r_val};"),
        ],
        MemWidth::U64 => {
            let r_val1 = reg(value + 1);
            vec![
                format!("cvt.u64.u32 %rd1, {r_addr};"),
                "add.u64 %rd1, %rd0, %rd1;".to_string(),
                format!("mov.b64 %rd2, {{{r_val}, {r_val1}}};"),
                "st.global.u64 [%rd1], %rd2;".to_string(),
            ]
        }
        MemWidth::U128 => {
            let r_v1 = reg(value + 1);
            let r_v2 = reg(value + 2);
            let r_v3 = reg(value + 3);
            vec![
                format!("cvt.u64.u32 %rd1, {r_addr};"),
                "add.u64 %rd1, %rd0, %rd1;".to_string(),
                format!("st.global.v4.u32 [%rd1], {{{r_val}, {r_v1}, {r_v2}, {r_v3}}};"),
            ]
        }
    }
}

#[must_use]
pub fn emit_shared_load(w: MemWidth, rd: u8, addr: u8) -> Vec<String> {
    let r_d = reg(rd);
    let r_addr = reg(addr);
    let pw = ptx_width(w);

    match w {
        MemWidth::U8 | MemWidth::U16 | MemWidth::U32 => vec![
            "mov.u32 %t0, _shared_mem;".to_string(),
            format!("add.u32 %t0, %t0, {r_addr};"),
            format!("ld.shared.{pw} {r_d}, [%t0];"),
        ],
        MemWidth::U64 => {
            let r_d1 = reg(rd + 1);
            vec![
                "mov.u32 %t0, _shared_mem;".to_string(),
                format!("add.u32 %t0, %t0, {r_addr};"),
                "ld.shared.u64 %rd2, [%t0];".to_string(),
                format!("mov.b64 {{{r_d}, {r_d1}}}, %rd2;"),
            ]
        }
        MemWidth::U128 => {
            let r_d1 = reg(rd + 1);
            let r_d2 = reg(rd + 2);
            let r_d3 = reg(rd + 3);
            vec![
                "mov.u32 %t0, _shared_mem;".to_string(),
                format!("add.u32 %t0, %t0, {r_addr};"),
                format!("ld.shared.v4.u32 {{{r_d}, {r_d1}, {r_d2}, {r_d3}}}, [%t0];"),
            ]
        }
    }
}

#[must_use]
pub fn emit_shared_store(w: MemWidth, addr: u8, value: u8) -> Vec<String> {
    let r_addr = reg(addr);
    let r_val = reg(value);
    let pw = ptx_width(w);

    match w {
        MemWidth::U8 | MemWidth::U16 | MemWidth::U32 => vec![
            "mov.u32 %t0, _shared_mem;".to_string(),
            format!("add.u32 %t0, %t0, {r_addr};"),
            format!("st.shared.{pw} [%t0], {r_val};"),
        ],
        MemWidth::U64 => {
            let r_v1 = reg(value + 1);
            vec![
                "mov.u32 %t0, _shared_mem;".to_string(),
                format!("add.u32 %t0, %t0, {r_addr};"),
                format!("mov.b64 %rd2, {{{r_val}, {r_v1}}};"),
                "st.shared.u64 [%t0], %rd2;".to_string(),
            ]
        }
        MemWidth::U128 => {
            let r_v1 = reg(value + 1);
            let r_v2 = reg(value + 2);
            let r_v3 = reg(value + 3);
            vec![
                "mov.u32 %t0, _shared_mem;".to_string(),
                format!("add.u32 %t0, %t0, {r_addr};"),
                format!(
                    "st.shared.v4.u32 [%t0], {{{r_val}, {r_v1}, {r_v2}, {r_v3}}};"
                ),
            ]
        }
    }
}

#[must_use]
fn atom_op_name(op: AtomicOp) -> &'static str {
    match op {
        AtomicOp::Add | AtomicOp::Sub => "add",
        AtomicOp::Min => "min",
        AtomicOp::Max => "max",
        AtomicOp::And => "and",
        AtomicOp::Or => "or",
        AtomicOp::Xor => "xor",
        AtomicOp::Exchange => "exch",
    }
}

#[must_use]
pub fn emit_global_atomic(
    op: AtomicOp,
    rd: Option<u8>,
    addr: u8,
    value: u8,
) -> Vec<String> {
    let r_addr = reg(addr);
    let r_val = reg(value);
    let dest = rd.map_or_else(|| "%t1".to_string(), reg);
    let op_name = atom_op_name(op);

    let mut lines = vec![
        format!("cvt.u64.u32 %rd1, {r_addr};"),
        "add.u64 %rd1, %rd0, %rd1;".to_string(),
    ];

    if op == AtomicOp::Sub {
        lines.push(format!("neg.s32 %t0, {r_val};"));
        lines.push(format!("atom.global.{op_name}.u32 {dest}, [%rd1], %t0;"));
    } else {
        lines.push(format!("atom.global.{op_name}.u32 {dest}, [%rd1], {r_val};"));
    }

    lines
}

#[must_use]
pub fn emit_shared_atomic(
    op: AtomicOp,
    rd: Option<u8>,
    addr: u8,
    value: u8,
) -> Vec<String> {
    let r_addr = reg(addr);
    let r_val = reg(value);
    let dest = rd.map_or_else(|| "%t1".to_string(), reg);
    let op_name = atom_op_name(op);

    let mut lines = vec![
        "mov.u32 %t0, _shared_mem;".to_string(),
        format!("add.u32 %t0, %t0, {r_addr};"),
    ];

    if op == AtomicOp::Sub {
        lines.push(format!("neg.s32 %t2, {r_val};"));
        lines.push(format!("atom.shared.{op_name}.u32 {dest}, [%t0], %t2;"));
    } else {
        lines.push(format!("atom.shared.{op_name}.u32 {dest}, [%t0], {r_val};"));
    }

    lines
}

#[must_use]
pub fn emit_global_atomic_cas(
    rd: Option<u8>,
    addr: u8,
    expected: u8,
    desired: u8,
) -> Vec<String> {
    let r_addr = reg(addr);
    let r_exp = reg(expected);
    let r_des = reg(desired);
    let dest = rd.map_or_else(|| "%t1".to_string(), reg);

    vec![
        format!("cvt.u64.u32 %rd1, {r_addr};"),
        "add.u64 %rd1, %rd0, %rd1;".to_string(),
        format!("atom.global.cas.b32 {dest}, [%rd1], {r_exp}, {r_des};"),
    ]
}

#[must_use]
pub fn emit_shared_atomic_cas(
    rd: Option<u8>,
    addr: u8,
    expected: u8,
    desired: u8,
) -> Vec<String> {
    let r_addr = reg(addr);
    let r_exp = reg(expected);
    let r_des = reg(desired);
    let dest = rd.map_or_else(|| "%t1".to_string(), reg);

    vec![
        "mov.u32 %t0, _shared_mem;".to_string(),
        format!("add.u32 %t0, %t0, {r_addr};"),
        format!("atom.shared.cas.b32 {dest}, [%t0], {r_exp}, {r_des};"),
    ]
}
