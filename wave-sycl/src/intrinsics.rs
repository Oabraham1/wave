// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! SYCL sub-group operation emission. Maps WAVE wave-level operations to standard
//!
//! SYCL 2020 sub-group functions: `select_from_group` for shuffle, `shift_group_left`/right
//! for `shuffle_down`/up, `permute_group_by_xor` for `shuffle_xor`, `group_broadcast` for
//! broadcast, `reduce_over_group` for reductions, `exclusive_scan_over_group` for prefix
//! sum, and `any_of_group`/`all_of_group` for vote operations. Sub-group width adapts
//! automatically (8 or 16 on Intel, 32 on NVIDIA via SYCL adaptor).

use crate::registers::{pred, reg};
use wave_decode::opcodes::{WaveOpType, WaveReduceType};

#[must_use]
#[allow(clippy::similar_names)]
pub fn emit_wave_op(op: WaveOpType, rd: u8, rs1: u8, rs2: Option<u8>) -> String {
    let r_d = reg(rd);
    let r_s1 = reg(rs1);
    let r_s2 = rs2.map(reg).unwrap_or_default();

    match op {
        WaveOpType::Shuffle => {
            format!("{r_d} = select_from_group(sg, {r_s1}, {r_s2});")
        }
        WaveOpType::ShuffleUp => {
            format!("{r_d} = shift_group_right(sg, {r_s1}, {r_s2});")
        }
        WaveOpType::ShuffleDown => {
            format!("{r_d} = shift_group_left(sg, {r_s1}, {r_s2});")
        }
        WaveOpType::ShuffleXor => {
            format!("{r_d} = permute_group_by_xor(sg, {r_s1}, {r_s2});")
        }
        WaveOpType::Broadcast => {
            format!("{r_d} = group_broadcast(sg, {r_s1}, {r_s2});")
        }
        WaveOpType::Ballot | WaveOpType::Any | WaveOpType::All => String::new(),
    }
}

#[must_use]
#[allow(clippy::similar_names)]
pub fn emit_wave_reduce(op: WaveReduceType, rd: u8, rs1: u8) -> String {
    let r_d = reg(rd);
    let r_s1 = reg(rs1);

    match op {
        WaveReduceType::PrefixSum => {
            format!("{r_d} = exclusive_scan_over_group(sg, {r_s1}, plus<uint32_t>());")
        }
        WaveReduceType::ReduceAdd => {
            format!("{r_d} = reduce_over_group(sg, {r_s1}, plus<uint32_t>());")
        }
        WaveReduceType::ReduceMin => {
            format!("{r_d} = (uint32_t)reduce_over_group(sg, (int32_t){r_s1}, minimum<int32_t>());")
        }
        WaveReduceType::ReduceMax => {
            format!("{r_d} = (uint32_t)reduce_over_group(sg, (int32_t){r_s1}, maximum<int32_t>());")
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
