// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Emit Metal shuffle, vote, reduction, and prefix-sum operations.

use crate::registers::{pred, reg};
use wave_decode::opcodes::{WaveOpType, WaveReduceType};

#[must_use]
pub fn emit_wave_op(op: WaveOpType, rd: u8, rs1: u8, rs2: Option<u8>) -> String {
    let r_d = reg(rd);
    let source1 = reg(rs1);
    let source2 = rs2.map(reg).unwrap_or_default();

    match op {
        WaveOpType::Shuffle => format!("{r_d} = simd_shuffle({source1}, {source2});"),
        WaveOpType::ShuffleUp => format!("{r_d} = simd_shuffle_up({source1}, {source2});"),
        WaveOpType::ShuffleDown => format!("{r_d} = simd_shuffle_down({source1}, {source2});"),
        WaveOpType::ShuffleXor => format!("{r_d} = simd_shuffle_xor({source1}, {source2});"),
        WaveOpType::Broadcast => format!("{r_d} = simd_broadcast({source1}, {source2});"),
        WaveOpType::Ballot | WaveOpType::Any | WaveOpType::All => String::new(),
    }
}

#[must_use]
pub fn emit_wave_reduce(op: WaveReduceType, rd: u8, rs1: u8) -> String {
    let r_d = reg(rd);
    let source1 = reg(rs1);

    match op {
        WaveReduceType::PrefixSum => format!("{r_d} = simd_prefix_exclusive_sum({source1});"),
        WaveReduceType::ReduceAdd => format!("{r_d} = simd_sum({source1});"),
        WaveReduceType::ReduceMin => format!("{r_d} = simd_min({source1});"),
        WaveReduceType::ReduceMax => format!("{r_d} = simd_max({source1});"),
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
