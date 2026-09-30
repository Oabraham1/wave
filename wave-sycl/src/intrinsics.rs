// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Emit SYCL C++ shuffle, vote, reduction, and prefix-sum operations.

use crate::registers::{pred, reg};
use wave_decode::opcodes::{WaveOpType, WaveReduceType};

#[must_use]
pub fn emit_wave_op(op: WaveOpType, rd: u8, rs1: u8, rs2: Option<u8>) -> String {
    let r_d = reg(rd);
    let source1 = reg(rs1);
    let source2 = rs2.map(reg).unwrap_or_default();

    match op {
        WaveOpType::Shuffle => {
            format!("{r_d} = select_from_group(sg, {source1}, {source2});")
        }
        WaveOpType::ShuffleUp => {
            format!("{r_d} = shift_group_right(sg, {source1}, {source2});")
        }
        WaveOpType::ShuffleDown => {
            format!("{r_d} = shift_group_left(sg, {source1}, {source2});")
        }
        WaveOpType::ShuffleXor => {
            format!("{r_d} = permute_group_by_xor(sg, {source1}, {source2});")
        }
        WaveOpType::Broadcast => {
            format!("{r_d} = group_broadcast(sg, {source1}, {source2});")
        }
        WaveOpType::Ballot | WaveOpType::Any | WaveOpType::All => String::new(),
    }
}

#[must_use]
pub fn emit_wave_reduce(op: WaveReduceType, rd: u8, rs1: u8) -> String {
    let r_d = reg(rd);
    let source1 = reg(rs1);

    match op {
        WaveReduceType::PrefixSum => {
            format!("{r_d} = exclusive_scan_over_group(sg, {source1}, plus<uint32_t>());")
        }
        WaveReduceType::ReduceAdd => {
            format!("{r_d} = reduce_over_group(sg, {source1}, plus<uint32_t>());")
        }
        WaveReduceType::ReduceMin => {
            format!(
                "{r_d} = (uint32_t)reduce_over_group(sg, (int32_t){source1}, minimum<int32_t>());"
            )
        }
        WaveReduceType::ReduceMax => {
            format!(
                "{r_d} = (uint32_t)reduce_over_group(sg, (int32_t){source1}, maximum<int32_t>());"
            )
        }
    }
}

#[must_use]
pub fn emit_wave_ballot(rd: u8, ps: u8) -> String {
    format!(
        "{} = reduce_over_group(sg, (uint32_t)({} ? (1u << sg.get_local_id()[0]) : 0u), bit_or<uint32_t>());",
        reg(rd),
        pred(ps)
    )
}

#[must_use]
pub fn emit_wave_vote(op: WaveOpType, pd: u8, ps: u8) -> String {
    let p_d = pred(pd);
    let p_s = pred(ps);

    match op {
        WaveOpType::Any => format!("{p_d} = any_of_group(sg, {p_s});"),
        WaveOpType::All => format!("{p_d} = all_of_group(sg, {p_s});"),
        _ => String::new(),
    }
}
