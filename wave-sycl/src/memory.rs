// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! SYCL memory operation emission. Generates load, store, and atomic operations for
//!
//! global (USM pointer device_mem_usm) and local (SLM pointer lm) memory spaces.
//! Loads and stores use byte-offset pointer arithmetic with C-style casts. Atomics
//! use sycl::atomic_ref with appropriate memory_order, memory_scope, and
//! address_space template parameters for each operation.

use crate::registers::reg;
use wave_decode::opcodes::{AtomicOp, MemWidth};

#[must_use]
pub fn c_type(width: MemWidth) -> &'static str {
    match width {
        MemWidth::U8 => "uint8_t",
        MemWidth::U16 => "uint16_t",
        MemWidth::U32 | MemWidth::U128 => "uint32_t",
        MemWidth::U64 => "uint64_t",
    }
}

#[must_use]
pub fn emit_load(mem_name: &str, width: MemWidth, rd: u8, addr: u8) -> Vec<String> {
    let r_d = reg(rd);
    let r_addr = reg(addr);
    let ty = c_type(width);

    match width {
        MemWidth::U8 | MemWidth::U16 | MemWidth::U32 => {
            vec![format!(
                "{r_d} = (uint32_t)(*({ty}*)({mem_name} + {r_addr}));"
            )]
        }
        MemWidth::U64 => {
            let r_d1 = reg(rd + 1);
            vec![
                "{".to_string(),
                format!("    uint64_t _tmp = *(uint64_t*)({mem_name} + {r_addr});"),
                format!("    {r_d} = (uint32_t)(_tmp & 0xFFFFFFFFu);"),
                format!("    {r_d1} = (uint32_t)(_tmp >> 32);"),
                "}".to_string(),
            ]
        }
        MemWidth::U128 => {
            let r_d1 = reg(rd + 1);
            let r_d2 = reg(rd + 2);
            let r_d3 = reg(rd + 3);
            vec![
                "{".to_string(),
                format!("    uint32_t* _p = (uint32_t*)({mem_name} + {r_addr});"),
                format!("    {r_d} = _p[0]; {r_d1} = _p[1]; {r_d2} = _p[2]; {r_d3} = _p[3];"),
                "}".to_string(),
            ]
        }
    }
}

#[must_use]
pub fn emit_store(mem_name: &str, width: MemWidth, addr: u8, value: u8) -> Vec<String> {
    let r_addr = reg(addr);
    let r_val = reg(value);
    let ty = c_type(width);

    match width {
        MemWidth::U8 | MemWidth::U16 | MemWidth::U32 => {
            vec![format!("*({ty}*)({mem_name} + {r_addr}) = ({ty}){r_val};")]
        }
        MemWidth::U64 => {
            let r_v1 = reg(value + 1);
            vec![format!(
                "*(uint64_t*)({mem_name} + {r_addr}) = (uint64_t){r_val} | ((uint64_t){r_v1} << 32);"
            )]
        }
        MemWidth::U128 => {
            let r_v1 = reg(value + 1);
            let r_v2 = reg(value + 2);
            let r_v3 = reg(value + 3);
            vec![
                "{".to_string(),
                format!("    uint32_t* _p = (uint32_t*)({mem_name} + {r_addr});"),
                format!("    _p[0] = {r_val}; _p[1] = {r_v1}; _p[2] = {r_v2}; _p[3] = {r_v3};"),
                "}".to_string(),
            ]
        }
    }
}

fn atomic_method(op: AtomicOp) -> &'static str {
    match op {
        AtomicOp::Add => "fetch_add",
        AtomicOp::Sub => "fetch_sub",
        AtomicOp::Min => "fetch_min",
        AtomicOp::Max => "fetch_max",
        AtomicOp::And => "fetch_and",
        AtomicOp::Or => "fetch_or",
        AtomicOp::Xor => "fetch_xor",
        AtomicOp::Exchange => "exchange",
    }
}

fn address_space(mem_name: &str) -> &'static str {
    if mem_name == "lm" {
        "access::address_space::local_space"
    } else {
        "access::address_space::global_space"
    }
}

fn scope_for_mem(mem_name: &str) -> &'static str {
    if mem_name == "lm" {
        "memory_scope::work_group"
    } else {
        "memory_scope::device"
    }
}

#[must_use]
pub fn emit_atomic(
    mem_name: &str,
    op: AtomicOp,
    rd: Option<u8>,
    addr: u8,
    value: u8,
) -> Vec<String> {
    let r_addr = reg(addr);
    let r_val = reg(value);
    let method = atomic_method(op);
    let aspace = address_space(mem_name);
    let scope = scope_for_mem(mem_name);

    let mut lines = vec![
        "{".to_string(),
        format!(
            "    atomic_ref<uint32_t, memory_order::relaxed, {scope}, {aspace}> _ar(*(uint32_t*)({mem_name} + {r_addr}));"
        ),
    ];

    match rd {
        Some(d) => lines.push(format!("    {} = _ar.{method}({r_val});", reg(d))),
        None => lines.push(format!("    _ar.{method}({r_val});")),
    }

    lines.push("}".to_string());
    lines
}

#[must_use]
pub fn emit_atomic_cas(
    mem_name: &str,
    rd: Option<u8>,
    addr: u8,
    expected: u8,
    desired: u8,
) -> Vec<String> {
    let r_addr = reg(addr);
    let r_exp = reg(expected);
    let r_des = reg(desired);
    let aspace = address_space(mem_name);
    let scope = scope_for_mem(mem_name);

    let mut lines = vec![
        "{".to_string(),
        format!(
            "    atomic_ref<uint32_t, memory_order::relaxed, {scope}, {aspace}> _ar(*(uint32_t*)({mem_name} + {r_addr}));"
        ),
        format!("    uint32_t _exp = {r_exp};"),
        format!("    _ar.compare_exchange_weak(_exp, {r_des});"),
    ];

    if let Some(d) = rd {
        lines.push(format!("    {} = _exp;", reg(d)));
    }

    lines.push("}".to_string());
    lines
}
