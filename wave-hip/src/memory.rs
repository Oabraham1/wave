// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! HIP memory operation emission. Generates load, store, and atomic operations for
//!
//! both global (device) and shared (local/LDS) memory. Memory accesses use byte-offset
//! pointer arithmetic with C-style casts. Atomics use HIP's atomicAdd/atomicSub/etc.
//! functions which work on both global and shared memory without address space qualifiers.

use crate::registers::reg;
use wave_decode::opcodes::{AtomicOp, MemWidth};

#[must_use]
pub fn msl_type(width: MemWidth) -> &'static str {
    match width {
        MemWidth::U8 => "uint8_t",
        MemWidth::U16 => "uint16_t",
        MemWidth::U32 => "uint32_t",
        MemWidth::U64 => "uint64_t",
        MemWidth::U128 => "uint4",
    }
}

#[must_use]
pub fn emit_load(mem_name: &str, width: MemWidth, rd: u8, addr: u8) -> Vec<String> {
    let r_d = reg(rd);
    let r_addr = reg(addr);
    let ty = msl_type(width);

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
                format!("    uint4 _tmp = *(uint4*)({mem_name} + {r_addr});"),
                format!("    {r_d} = _tmp.x;"),
                format!("    {r_d1} = _tmp.y;"),
                format!("    {r_d2} = _tmp.z;"),
                format!("    {r_d3} = _tmp.w;"),
                "}".to_string(),
            ]
        }
    }
}

#[must_use]
pub fn emit_store(mem_name: &str, width: MemWidth, addr: u8, value: u8) -> Vec<String> {
    let r_addr = reg(addr);
    let r_val = reg(value);
    let ty = msl_type(width);

    match width {
        MemWidth::U8 | MemWidth::U16 | MemWidth::U32 => {
            vec![format!("*({ty}*)({mem_name} + {r_addr}) = ({ty}){r_val};")]
        }
        MemWidth::U64 => {
            let r_val1 = reg(value + 1);
            vec![format!(
                "*(uint64_t*)({mem_name} + {r_addr}) = (uint64_t){r_val} | ((uint64_t){r_val1} << 32);"
            )]
        }
        MemWidth::U128 => {
            let r_v1 = reg(value + 1);
            let r_v2 = reg(value + 2);
            let r_v3 = reg(value + 3);
            vec![format!(
                "*(uint4*)({mem_name} + {r_addr}) = make_uint4({r_val}, {r_v1}, {r_v2}, {r_v3});"
            )]
        }
    }
}

#[must_use]
fn atomic_func(op: AtomicOp) -> &'static str {
    match op {
        AtomicOp::Add => "atomicAdd",
        AtomicOp::Sub => "atomicSub",
        AtomicOp::Min => "atomicMin",
        AtomicOp::Max => "atomicMax",
        AtomicOp::And => "atomicAnd",
        AtomicOp::Or => "atomicOr",
        AtomicOp::Xor => "atomicXor",
        AtomicOp::Exchange => "atomicExch",
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
    let func = atomic_func(op);

    match rd {
        Some(d) => {
            vec![format!(
                "{} = {}((uint32_t*)({mem_name} + {r_addr}), {r_val});",
                reg(d),
                func
            )]
        }
        None => {
            vec![format!(
                "{}((uint32_t*)({mem_name} + {r_addr}), {r_val});",
                func
            )]
        }
    }
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

    match rd {
        Some(d) => {
            vec![format!(
                "{} = atomicCAS((uint32_t*)({mem_name} + {r_addr}), {r_exp}, {r_des});",
                reg(d)
            )]
        }
        None => {
            vec![format!(
                "atomicCAS((uint32_t*)({mem_name} + {r_addr}), {r_exp}, {r_des});"
            )]
        }
    }
}
