// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Emit PTX shuffle, vote, reduction, and prefix-sum operations.

use crate::registers::{pred, reg};
use wave_decode::opcodes::{WaveOpType, WaveReduceType};

const FULL_MASK: &str = "0xFFFFFFFF";

#[must_use]
pub fn emit_wave_op(op: WaveOpType, rd: u8, rs1: u8, rs2: Option<u8>) -> Vec<String> {
    let r_d = reg(rd);
    let source1 = reg(rs1);
    let source2 = rs2.map_or_else(|| "0".to_string(), reg);

    match op {
        WaveOpType::Shuffle | WaveOpType::Broadcast => vec![format!(
            "shfl.sync.idx.b32 {r_d}, {source1}, {source2}, 31, {FULL_MASK};"
        )],
        WaveOpType::ShuffleUp => vec![format!(
            "shfl.sync.up.b32 {r_d}, {source1}, {source2}, 0, {FULL_MASK};"
        )],
        WaveOpType::ShuffleDown => vec![format!(
            "shfl.sync.down.b32 {r_d}, {source1}, {source2}, 31, {FULL_MASK};"
        )],
        WaveOpType::ShuffleXor => vec![format!(
            "shfl.sync.bfly.b32 {r_d}, {source1}, {source2}, 31, {FULL_MASK};"
        )],
        WaveOpType::Ballot | WaveOpType::Any | WaveOpType::All => Vec::new(),
    }
}

#[must_use]
pub fn emit_wave_ballot(rd: u8, ps: u8) -> Vec<String> {
    vec![format!(
        "vote.sync.ballot.b32 {}, {}, {FULL_MASK};",
        reg(rd),
        pred(ps)
    )]
}

#[must_use]
pub fn emit_wave_vote(op: WaveOpType, pd: u8, ps: u8) -> Vec<String> {
    let p_d = pred(pd);
    let p_s = pred(ps);

    match op {
        WaveOpType::Any => vec![format!("vote.sync.any.pred {p_d}, {p_s}, {FULL_MASK};")],
        WaveOpType::All => vec![format!("vote.sync.all.pred {p_d}, {p_s}, {FULL_MASK};")],
        _ => Vec::new(),
    }
}

fn emit_shfl_reduce(rd: u8, rs1: u8, combine_op: &str) -> Vec<String> {
    let r_d = reg(rd);
    let source1 = reg(rs1);
    let mut lines = vec![format!("mov.b32 {r_d}, {source1};")];

    for delta in [16, 8, 4, 2, 1] {
        lines.push(format!(
            "shfl.sync.bfly.b32 %t0, {r_d}, {delta}, 31, {FULL_MASK};"
        ));
        lines.push(format!("{combine_op} {r_d}, {r_d}, %t0;"));
    }

    lines
}

#[must_use]
pub fn emit_wave_reduce(op: WaveReduceType, rd: u8, rs1: u8) -> Vec<String> {
    match op {
        WaveReduceType::ReduceAdd => emit_shfl_reduce(rd, rs1, "add.s32"),
        WaveReduceType::ReduceMin => emit_shfl_reduce(rd, rs1, "min.s32"),
        WaveReduceType::ReduceMax => emit_shfl_reduce(rd, rs1, "max.s32"),
        WaveReduceType::PrefixSum => emit_prefix_sum(rd, rs1),
    }
}

fn emit_prefix_sum(rd: u8, rs1: u8) -> Vec<String> {
    let r_d = reg(rd);
    let source1 = reg(rs1);
    let mut lines = vec![format!("mov.b32 {r_d}, {source1};")];

    for delta in [1, 2, 4, 8, 16] {
        lines.push(format!(
            "shfl.sync.up.b32 %t0|%pt0, {r_d}, {delta}, 0, {FULL_MASK};"
        ));
        lines.push(format!("@%pt0 add.s32 {r_d}, {r_d}, %t0;"));
    }

    lines.push(format!("mov.b32 %t0, {r_d};"));
    lines.push(format!(
        "shfl.sync.up.b32 {r_d}|%pt0, %t0, 1, 0, {FULL_MASK};"
    ));
    lines.push(format!("@!%pt0 mov.b32 {r_d}, 0;"));

    lines
}
