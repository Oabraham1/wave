// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Wave (SIMD-group) operation MSL emission. Maps WAVE wave-level operations to
//!
//! Metal's native simd_* intrinsic functions. Apple Silicon GPUs have 32-wide SIMD
//! groups and Metal provides direct intrinsics for shuffle, broadcast, ballot,
//! vote, prefix sum, and reductions through the `metal_simdgroup` header.

use crate::registers::{pred, reg};
use wave_decode::opcodes::{WaveOpType, WaveReduceType};

#[must_use]
#[allow(clippy::similar_names)]
pub fn emit_wave_op(op: WaveOpType, rd: u8, rs1: u8, rs2: Option<u8>) -> String {
    let r_d = reg(rd);
    let r_s1 = reg(rs1);
    let r_s2 = rs2.map(reg).unwrap_or_default();

    match op {
        WaveOpType::Shuffle => format!("{r_d} = simd_shuffle({r_s1}, {r_s2});"),
        WaveOpType::ShuffleUp => format!("{r_d} = simd_shuffle_up({r_s1}, {r_s2});"),
        WaveOpType::ShuffleDown => format!("{r_d} = simd_shuffle_down({r_s1}, {r_s2});"),
        WaveOpType::ShuffleXor => format!("{r_d} = simd_shuffle_xor({r_s1}, {r_s2});"),
        WaveOpType::Broadcast => format!("{r_d} = simd_broadcast({r_s1}, {r_s2});"),
        WaveOpType::Ballot | WaveOpType::Any | WaveOpType::All => String::new(),
    }
}

#[must_use]
#[allow(clippy::similar_names)]
pub fn emit_wave_reduce(op: WaveReduceType, rd: u8, rs1: u8) -> String {
    let r_d = reg(rd);
    let r_s1 = reg(rs1);

    match op {
        WaveReduceType::PrefixSum => format!("{r_d} = simd_prefix_exclusive_sum({r_s1});"),
        WaveReduceType::ReduceAdd => format!("{r_d} = simd_sum({r_s1});"),
        WaveReduceType::ReduceMin => format!("{r_d} = simd_min({r_s1});"),
        WaveReduceType::ReduceMax => format!("{r_d} = simd_max({r_s1});"),
    }
}

#[must_use]
pub fn emit_wave_ballot(rd: u8, ps: u8) -> String {
    let r_d = reg(rd);
    let p_s = pred(ps);
    format!("{r_d} = (uint32_t)simd_ballot({p_s});")
}

#[must_use]
pub fn emit_wave_vote(op: WaveOpType, pd: u8, ps: u8) -> String {
    let p_d = pred(pd);
    let p_s = pred(ps);

    match op {
        WaveOpType::Any => format!("{p_d} = simd_any({p_s});"),
        WaveOpType::All => format!("{p_d} = simd_all({p_s});"),
        _ => String::new(),
    }
}
