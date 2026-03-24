// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! AMD wavefront intrinsic emission for HIP. Maps WAVE wave-level operations to HIP
//!
//! __shfl/__shfl_up/__shfl_down/__shfl_xor intrinsics. Reductions use a shfl-based
//! butterfly tree with log2(warpSize) iterations, handling both RDNA (wavefront 32)
//! and CDNA (wavefront 64) via the warpSize runtime variable. Prefix sum uses
//! shfl_up with predicated accumulation.

use crate::registers::{pred, reg};
use wave_decode::opcodes::{WaveOpType, WaveReduceType};

#[must_use]
#[allow(clippy::similar_names)]
pub fn emit_wave_op(op: WaveOpType, rd: u8, rs1: u8, rs2: Option<u8>) -> String {
    let r_d = reg(rd);
    let r_s1 = reg(rs1);
    let r_s2 = rs2.map(reg).unwrap_or_default();

    match op {
        WaveOpType::Shuffle | WaveOpType::Broadcast => {
            format!("{r_d} = (uint32_t)__shfl((int){r_s1}, {r_s2});")
        }
        WaveOpType::ShuffleUp => {
            format!("{r_d} = (uint32_t)__shfl_up((int){r_s1}, {r_s2});")
        }
        WaveOpType::ShuffleDown => {
            format!("{r_d} = (uint32_t)__shfl_down((int){r_s1}, {r_s2});")
        }
        WaveOpType::ShuffleXor => {
            format!("{r_d} = (uint32_t)__shfl_xor((int){r_s1}, {r_s2});")
        }
        WaveOpType::Ballot | WaveOpType::Any | WaveOpType::All => String::new(),
    }
}

#[must_use]
#[allow(clippy::similar_names)]
pub fn emit_wave_reduce(op: WaveReduceType, rd: u8, rs1: u8) -> Vec<String> {
    let r_d = reg(rd);
    let r_s1 = reg(rs1);

    match op {
        WaveReduceType::PrefixSum => emit_prefix_sum(&r_d, &r_s1),
        WaveReduceType::ReduceAdd => emit_shfl_reduce(&r_d, &r_s1, "+="),
        WaveReduceType::ReduceMin => emit_reduce_minmax(&r_d, &r_s1, "min"),
        WaveReduceType::ReduceMax => emit_reduce_minmax(&r_d, &r_s1, "max"),
    }
}

#[must_use]
pub fn emit_wave_ballot(rd: u8, ps: u8) -> String {
    format!(
        "{} = (uint32_t)__ballot((int){});",
        reg(rd),
        pred(ps)
    )
}

#[must_use]
pub fn emit_wave_vote(op: WaveOpType, pd: u8, ps: u8) -> String {
    let p_d = pred(pd);
    let p_s = pred(ps);

    match op {
        WaveOpType::Any => format!("{p_d} = __any((int){p_s});"),
        WaveOpType::All => format!("{p_d} = __all((int){p_s});"),
        _ => String::new(),
    }
}

fn emit_shfl_reduce(r_d: &str, r_s: &str, combine_op: &str) -> Vec<String> {
    vec![
        "{".to_string(),
        format!("    uint32_t _rv = {r_s};"),
        format!("    for (int _off = warpSize >> 1; _off > 0; _off >>= 1)"),
        format!("        _rv {combine_op} (uint32_t)__shfl_down((int)_rv, _off);"),
        format!("    {r_d} = (uint32_t)__shfl((int)_rv, 0);"),
        "}".to_string(),
    ]
}

fn emit_reduce_minmax(r_d: &str, r_s: &str, func: &str) -> Vec<String> {
    vec![
        "{".to_string(),
        format!("    int32_t _rv = (int32_t){r_s};"),
        "    for (int _off = warpSize >> 1; _off > 0; _off >>= 1)".to_string(),
        format!("        _rv = {func}(_rv, __shfl_down(_rv, _off));"),
        format!("    {r_d} = (uint32_t)__shfl(_rv, 0);"),
        "}".to_string(),
    ]
}

fn emit_prefix_sum(r_d: &str, r_s: &str) -> Vec<String> {
    vec![
        "{".to_string(),
        format!("    uint32_t _pv = {r_s};"),
        "    for (int _off = 1; _off < warpSize; _off <<= 1) {".to_string(),
        "        uint32_t _tmp = (uint32_t)__shfl_up((int)_pv, _off);".to_string(),
        "        if (__lane_id() >= (uint32_t)_off) _pv += _tmp;".to_string(),
        "    }".to_string(),
        "    uint32_t _incl = _pv;".to_string(),
        "    _pv = (uint32_t)__shfl_up((int)_incl, 1);".to_string(),
        format!("    {r_d} = (__lane_id() == 0) ? 0u : _pv;"),
        "}".to_string(),
    ]
}
