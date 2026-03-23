// Copyright (c) 2026 Ojima Abraham. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

// Instruction-to-text formatter. Converts DecodedInstruction from wave-decode
// into human-readable WAVE assembly text. Handles all instruction variants,
// predication, register naming, and immediate formatting.

use wave_decode::{
    special_register_name, AtomicOp, BitOpType, CmpOp, CvtType, DecodedInstruction, F16Op,
    F16PackedOp, F64DivSqrtOp, F64Op, FUnaryOp, MemWidth, Operation, Scope, WaveOpType,
    WaveReduceType,
};

#[derive(Default)]
pub struct FormatOptions {
    pub show_offsets: bool,
    pub show_raw: bool,
}

pub struct FormattedLine {
    pub offset: u32,
    pub raw_bytes: Vec<u8>,
    pub text: String,
}

#[must_use]
pub fn format_instruction(inst: &DecodedInstruction) -> String {
    let predicate_prefix = if inst.predicate != 0 {
        if inst.predicate_negated {
            format!("@!p{} ", inst.predicate)
        } else {
            format!("@p{} ", inst.predicate)
        }
    } else {
        String::new()
    };

    format!("{}{}", predicate_prefix, format_operation(&inst.operation))
}

#[allow(clippy::too_many_lines)]
fn format_operation(op: &Operation) -> String {
    match op {
        Operation::Iadd { rd, rs1, rs2 } => format!("iadd r{rd}, r{rs1}, r{rs2}"),
        Operation::Isub { rd, rs1, rs2 } => format!("isub r{rd}, r{rs1}, r{rs2}"),
        Operation::Imul { rd, rs1, rs2 } => format!("imul r{rd}, r{rs1}, r{rs2}"),
        Operation::ImulHi { rd, rs1, rs2 } => format!("imul_hi r{rd}, r{rs1}, r{rs2}"),
        Operation::Imad { rd, rs1, rs2, rs3 } => format!("imad r{rd}, r{rs1}, r{rs2}, r{rs3}"),
        Operation::Idiv { rd, rs1, rs2 } => format!("idiv r{rd}, r{rs1}, r{rs2}"),
        Operation::Imod { rd, rs1, rs2 } => format!("imod r{rd}, r{rs1}, r{rs2}"),
        Operation::Ineg { rd, rs1 } => format!("ineg r{rd}, r{rs1}"),
        Operation::Iabs { rd, rs1 } => format!("iabs r{rd}, r{rs1}"),
        Operation::Imin { rd, rs1, rs2 } => format!("imin r{rd}, r{rs1}, r{rs2}"),
        Operation::Imax { rd, rs1, rs2 } => format!("imax r{rd}, r{rs1}, r{rs2}"),
        Operation::Iclamp { rd, rs1, rs2, rs3 } => format!("iclamp r{rd}, r{rs1}, r{rs2}, r{rs3}"),

        Operation::Fadd { rd, rs1, rs2 } => format!("fadd r{rd}, r{rs1}, r{rs2}"),
        Operation::Fsub { rd, rs1, rs2 } => format!("fsub r{rd}, r{rs1}, r{rs2}"),
        Operation::Fmul { rd, rs1, rs2 } => format!("fmul r{rd}, r{rs1}, r{rs2}"),
        Operation::Fma { rd, rs1, rs2, rs3 } => format!("fma r{rd}, r{rs1}, r{rs2}, r{rs3}"),
        Operation::Fdiv { rd, rs1, rs2 } => format!("fdiv r{rd}, r{rs1}, r{rs2}"),
        Operation::Fneg { rd, rs1 } => format!("fneg r{rd}, r{rs1}"),
        Operation::Fabs { rd, rs1 } => format!("fabs r{rd}, r{rs1}"),
        Operation::Fmin { rd, rs1, rs2 } => format!("fmin r{rd}, r{rs1}, r{rs2}"),
        Operation::Fmax { rd, rs1, rs2 } => format!("fmax r{rd}, r{rs1}, r{rs2}"),
        Operation::Fclamp { rd, rs1, rs2, rs3 } => format!("fclamp r{rd}, r{rs1}, r{rs2}, r{rs3}"),
        Operation::Fsqrt { rd, rs1 } => format!("fsqrt r{rd}, r{rs1}"),
        Operation::FUnary { op, rd, rs1 } => format!("{} r{rd}, r{rs1}", funary_mnemonic(*op)),

        Operation::F16 { op, rd, rs1, rs2, rs3 } => {
            if let Some(rs3) = rs3 {
                format!("{} r{rd}, r{rs1}, r{rs2}, r{rs3}", f16_mnemonic(*op))
            } else {
                format!("{} r{rd}, r{rs1}, r{rs2}", f16_mnemonic(*op))
            }
        }
        Operation::F16Packed { op, rd, rs1, rs2, rs3 } => {
            if let Some(rs3) = rs3 {
                format!("{} r{rd}, r{rs1}, r{rs2}, r{rs3}", f16_packed_mnemonic(*op))
            } else {
                format!("{} r{rd}, r{rs1}, r{rs2}", f16_packed_mnemonic(*op))
            }
        }
        Operation::F64 { op, rd, rs1, rs2, rs3 } => {
            if let Some(rs3) = rs3 {
                format!("{} r{rd}, r{rs1}, r{rs2}, r{rs3}", f64_mnemonic(*op))
            } else {
                format!("{} r{rd}, r{rs1}, r{rs2}", f64_mnemonic(*op))
            }
        }
        Operation::F64DivSqrt { op, rd, rs1, rs2 } => {
            if let Some(rs2) = rs2 {
                format!("{} r{rd}, r{rs1}, r{rs2}", f64_divsqrt_mnemonic(*op))
            } else {
                format!("{} r{rd}, r{rs1}", f64_divsqrt_mnemonic(*op))
            }
        }

        Operation::And { rd, rs1, rs2 } => format!("and r{rd}, r{rs1}, r{rs2}"),
        Operation::Or { rd, rs1, rs2 } => format!("or r{rd}, r{rs1}, r{rs2}"),
        Operation::Xor { rd, rs1, rs2 } => format!("xor r{rd}, r{rs1}, r{rs2}"),
        Operation::Not { rd, rs1 } => format!("not r{rd}, r{rs1}"),
        Operation::Shl { rd, rs1, rs2 } => format!("shl r{rd}, r{rs1}, r{rs2}"),
        Operation::Shr { rd, rs1, rs2 } => format!("shr r{rd}, r{rs1}, r{rs2}"),
        Operation::Sar { rd, rs1, rs2 } => format!("sar r{rd}, r{rs1}, r{rs2}"),
        Operation::BitOp { op, rd, rs1, rs2, rs3, rs4 } => format_bitop(*op, *rd, *rs1, *rs2, *rs3, *rs4),

        Operation::Icmp { op, pd, rs1, rs2 } => {
            format!("icmp_{} p{pd}, r{rs1}, r{rs2}", cmp_suffix(*op))
        }
        Operation::Ucmp { op, pd, rs1, rs2 } => {
            format!("ucmp_{} p{pd}, r{rs1}, r{rs2}", cmp_suffix(*op))
        }
        Operation::Fcmp { op, pd, rs1, rs2 } => {
            format!("fcmp_{} p{pd}, r{rs1}, r{rs2}", cmp_suffix(*op))
        }

        Operation::Select { rd, ps, rs1, rs2 } => format!("select r{rd}, p{ps}, r{rs1}, r{rs2}"),
        Operation::Cvt { cvt_type, rd, rs1 } => format!("{} r{rd}, r{rs1}", cvt_mnemonic(*cvt_type)),

        Operation::LocalLoad { width, rd, addr } => {
            format!("local_load_{} r{rd}, r{addr}", width_suffix(*width))
        }
        Operation::LocalStore { width, addr, value } => {
            format!("local_store_{} r{addr}, r{value}", width_suffix(*width))
        }
        Operation::DeviceLoad { width, rd, addr } => {
            format!("device_load_{} r{rd}, r{addr}", width_suffix(*width))
        }
        Operation::DeviceStore { width, addr, value } => {
            format!("device_store_{} r{addr}, r{value}", width_suffix(*width))
        }

        Operation::LocalAtomic { op, rd, addr, value } => {
            if let Some(rd) = rd {
                format!("local_atomic_{} r{rd}, r{addr}, r{value}", atomic_suffix(*op))
            } else {
                format!("local_atomic_{} r{addr}, r{value}", atomic_suffix(*op))
            }
        }
        Operation::LocalAtomicCas { rd, addr, expected, desired } => {
            if let Some(rd) = rd {
                format!("local_atomic_cas r{rd}, r{addr}, r{expected}, r{desired}")
            } else {
                format!("local_atomic_cas r{addr}, r{expected}, r{desired}")
            }
        }
        Operation::DeviceAtomic { op, rd, addr, value, scope } => {
            let scope_str = scope_suffix(*scope);
            if let Some(rd) = rd {
                format!("atomic_{}.{scope_str} r{rd}, r{addr}, r{value}", atomic_suffix(*op))
            } else {
                format!("atomic_{}.{scope_str} r{addr}, r{value}", atomic_suffix(*op))
            }
        }
        Operation::DeviceAtomicCas { rd, addr, expected, desired, scope } => {
            let scope_str = scope_suffix(*scope);
            if let Some(rd) = rd {
                format!("atomic_cas.{scope_str} r{rd}, r{addr}, r{expected}, r{desired}")
            } else {
                format!("atomic_cas.{scope_str} r{addr}, r{expected}, r{desired}")
            }
        }

        Operation::WaveOp { op, rd, rs1, rs2 } => {
            if let Some(rs2) = rs2 {
                format!("{} r{rd}, r{rs1}, r{rs2}", wave_op_mnemonic(*op))
            } else {
                format!("{} r{rd}, r{rs1}", wave_op_mnemonic(*op))
            }
        }
        Operation::WaveReduce { op, rd, rs1 } => {
            format!("{} r{rd}, r{rs1}", wave_reduce_mnemonic(*op))
        }
        Operation::WaveBallot { rd, ps } => format!("wave_ballot r{rd}, p{ps}"),
        Operation::WaveVote { op, pd, ps } => format!("{} p{pd}, p{ps}", wave_op_mnemonic(*op)),

        Operation::If { ps } => format!("if p{ps}"),
        Operation::Else => "else".to_string(),
        Operation::Endif => "endif".to_string(),
        Operation::Loop => "loop".to_string(),
        Operation::Break { ps } => format!("break p{ps}"),
        Operation::Continue { ps } => format!("continue p{ps}"),
        Operation::Endloop => "endloop".to_string(),
        Operation::Call { target } => format!("call loc_{target:04X}"),

        Operation::Return => "return".to_string(),
        Operation::Halt => "halt".to_string(),
        Operation::Barrier => "barrier".to_string(),
        Operation::FenceAcquire { scope } => format!("fence_acquire.{}", scope_suffix(*scope)),
        Operation::FenceRelease { scope } => format!("fence_release.{}", scope_suffix(*scope)),
        Operation::FenceAcqRel { scope } => format!("fence_acq_rel.{}", scope_suffix(*scope)),
        Operation::Wait => "wait".to_string(),
        Operation::Nop => "nop".to_string(),

        Operation::Mov { rd, rs1 } => format!("mov r{rd}, r{rs1}"),
        Operation::MovImm { rd, imm } => {
            if *imm <= 9 {
                format!("mov_imm r{rd}, {imm}")
            } else {
                format!("mov_imm r{rd}, 0x{imm:X}")
            }
        }
        Operation::MovSr { rd, sr_index } => {
            let sr_name = special_register_name(*sr_index).unwrap_or("sr_unknown");
            format!("mov r{rd}, {sr_name}")
        }

        Operation::Unknown { opcode, word0, word1 } => {
            if let Some(w1) = word1 {
                format!(".unknown 0x{opcode:02X} ; {word0:08X} {w1:08X}")
            } else {
                format!(".unknown 0x{opcode:02X} ; {word0:08X}")
            }
        }
    }
}

fn funary_mnemonic(op: FUnaryOp) -> &'static str {
    match op {
        FUnaryOp::Frsqrt => "frsqrt",
        FUnaryOp::Frcp => "frcp",
        FUnaryOp::Ffloor => "ffloor",
        FUnaryOp::Fceil => "fceil",
        FUnaryOp::Fround => "fround",
        FUnaryOp::Ftrunc => "ftrunc",
        FUnaryOp::Ffract => "ffract",
        FUnaryOp::Fsat => "fsat",
        FUnaryOp::Fsin => "fsin",
        FUnaryOp::Fcos => "fcos",
        FUnaryOp::Fexp2 => "fexp2",
        FUnaryOp::Flog2 => "flog2",
    }
}

fn f16_mnemonic(op: F16Op) -> &'static str {
    match op {
        F16Op::Hadd => "hadd",
        F16Op::Hsub => "hsub",
        F16Op::Hmul => "hmul",
        F16Op::Hma => "hma",
    }
}

fn f16_packed_mnemonic(op: F16PackedOp) -> &'static str {
    match op {
        F16PackedOp::Hadd2 => "hadd2",
        F16PackedOp::Hmul2 => "hmul2",
        F16PackedOp::Hma2 => "hma2",
    }
}

fn f64_mnemonic(op: F64Op) -> &'static str {
    match op {
        F64Op::Dadd => "dadd",
        F64Op::Dsub => "dsub",
        F64Op::Dmul => "dmul",
        F64Op::Dma => "dma",
    }
}

fn f64_divsqrt_mnemonic(op: F64DivSqrtOp) -> &'static str {
    match op {
        F64DivSqrtOp::Ddiv => "ddiv",
        F64DivSqrtOp::Dsqrt => "dsqrt",
    }
}

fn format_bitop(op: BitOpType, rd: u8, rs1: u8, rs2: Option<u8>, rs3: Option<u8>, rs4: Option<u8>) -> String {
    match op {
        BitOpType::Bitcount => format!("bitcount r{rd}, r{rs1}"),
        BitOpType::Bitfind => format!("bitfind r{rd}, r{rs1}"),
        BitOpType::Bitrev => format!("bitrev r{rd}, r{rs1}"),
        BitOpType::Bfe => {
            let rs2 = rs2.unwrap_or(0);
            let rs3 = rs3.unwrap_or(0);
            format!("bfe r{rd}, r{rs1}, r{rs2}, r{rs3}")
        }
        BitOpType::Bfi => {
            let rs2 = rs2.unwrap_or(0);
            let rs3 = rs3.unwrap_or(0);
            let rs4 = rs4.unwrap_or(0);
            format!("bfi r{rd}, r{rs1}, r{rs2}, r{rs3}, r{rs4}")
        }
    }
}

fn cmp_suffix(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Eq => "eq",
        CmpOp::Ne => "ne",
        CmpOp::Lt => "lt",
        CmpOp::Le => "le",
        CmpOp::Gt => "gt",
        CmpOp::Ge => "ge",
        CmpOp::Ord => "ord",
        CmpOp::Unord => "unord",
    }
}

fn cvt_mnemonic(cvt_type: CvtType) -> &'static str {
    match cvt_type {
        CvtType::F32I32 => "cvt_f32_i32",
        CvtType::F32U32 => "cvt_f32_u32",
        CvtType::I32F32 => "cvt_i32_f32",
        CvtType::U32F32 => "cvt_u32_f32",
        CvtType::F32F16 => "cvt_f32_f16",
        CvtType::F16F32 => "cvt_f16_f32",
        CvtType::F32F64 => "cvt_f32_f64",
        CvtType::F64F32 => "cvt_f64_f32",
    }
}

fn width_suffix(width: MemWidth) -> &'static str {
    match width {
        MemWidth::U8 => "u8",
        MemWidth::U16 => "u16",
        MemWidth::U32 => "u32",
        MemWidth::U64 => "u64",
        MemWidth::U128 => "u128",
    }
}

fn atomic_suffix(op: AtomicOp) -> &'static str {
    match op {
        AtomicOp::Add => "add",
        AtomicOp::Sub => "sub",
        AtomicOp::Min => "min",
        AtomicOp::Max => "max",
        AtomicOp::And => "and",
        AtomicOp::Or => "or",
        AtomicOp::Xor => "xor",
        AtomicOp::Exchange => "exchange",
    }
}

fn scope_suffix(scope: Scope) -> &'static str {
    match scope {
        Scope::Wave => "wave",
        Scope::Workgroup => "workgroup",
        Scope::Device => "device",
        Scope::System => "system",
    }
}

fn wave_op_mnemonic(op: WaveOpType) -> &'static str {
    match op {
        WaveOpType::Shuffle => "wave_shuffle",
        WaveOpType::ShuffleUp => "wave_shuffle_up",
        WaveOpType::ShuffleDown => "wave_shuffle_down",
        WaveOpType::ShuffleXor => "wave_shuffle_xor",
        WaveOpType::Broadcast => "wave_broadcast",
        WaveOpType::Ballot => "wave_ballot",
        WaveOpType::Any => "wave_any",
        WaveOpType::All => "wave_all",
    }
}

fn wave_reduce_mnemonic(op: WaveReduceType) -> &'static str {
    match op {
        WaveReduceType::PrefixSum => "wave_prefix_sum",
        WaveReduceType::ReduceAdd => "wave_reduce_add",
        WaveReduceType::ReduceMin => "wave_reduce_min",
        WaveReduceType::ReduceMax => "wave_reduce_max",
    }
}

#[must_use]
pub fn format_with_offset(inst: &DecodedInstruction, code: &[u8], code_base: u32) -> String {
    let text = format_instruction(inst);
    let abs_offset = code_base + inst.offset;
    let start = inst.offset as usize;
    let end = start + inst.size as usize;
    let raw = &code[start..end.min(code.len())];

    let raw_hex = if inst.size == 8 && raw.len() >= 8 {
        let w0 = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        let w1 = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
        format!("0x{abs_offset:04X}: {w0:08X}  {text}\n        {w1:08X}")
    } else if raw.len() >= 4 {
        let w0 = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        format!("0x{abs_offset:04X}: {w0:08X}  {text}")
    } else {
        format!("0x{abs_offset:04X}: ????????  {text}")
    };

    raw_hex
}

#[must_use]
pub fn format_offset_only(inst: &DecodedInstruction, code_base: u32) -> String {
    let text = format_instruction(inst);
    let abs_offset = code_base + inst.offset;
    format!("0x{abs_offset:04X}: {text}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_iadd() {
        let inst = DecodedInstruction {
            offset: 0,
            size: 4,
            operation: Operation::Iadd { rd: 3, rs1: 1, rs2: 2 },
            predicate: 0,
            predicate_negated: false,
        };
        assert_eq!(format_instruction(&inst), "iadd r3, r1, r2");
    }

    #[test]
    fn test_format_mov_imm_small() {
        let inst = DecodedInstruction {
            offset: 0,
            size: 8,
            operation: Operation::MovImm { rd: 0, imm: 4 },
            predicate: 0,
            predicate_negated: false,
        };
        assert_eq!(format_instruction(&inst), "mov_imm r0, 4");
    }

    #[test]
    fn test_format_mov_imm_large() {
        let inst = DecodedInstruction {
            offset: 0,
            size: 8,
            operation: Operation::MovImm { rd: 7, imm: 0xDEADBEEF },
            predicate: 0,
            predicate_negated: false,
        };
        assert_eq!(format_instruction(&inst), "mov_imm r7, 0xDEADBEEF");
    }

    #[test]
    fn test_format_predicated() {
        let inst = DecodedInstruction {
            offset: 0,
            size: 4,
            operation: Operation::Iadd { rd: 3, rs1: 1, rs2: 2 },
            predicate: 1,
            predicate_negated: false,
        };
        assert_eq!(format_instruction(&inst), "@p1 iadd r3, r1, r2");
    }

    #[test]
    fn test_format_predicated_negated() {
        let inst = DecodedInstruction {
            offset: 0,
            size: 4,
            operation: Operation::Fadd { rd: 0, rs1: 1, rs2: 2 },
            predicate: 1,
            predicate_negated: true,
        };
        assert_eq!(format_instruction(&inst), "@!p1 fadd r0, r1, r2");
    }

    #[test]
    fn test_format_barrier() {
        let inst = DecodedInstruction {
            offset: 0,
            size: 4,
            operation: Operation::Barrier,
            predicate: 0,
            predicate_negated: false,
        };
        assert_eq!(format_instruction(&inst), "barrier");
    }

    #[test]
    fn test_format_halt() {
        let inst = DecodedInstruction {
            offset: 0,
            size: 4,
            operation: Operation::Halt,
            predicate: 0,
            predicate_negated: false,
        };
        assert_eq!(format_instruction(&inst), "halt");
    }

    #[test]
    fn test_format_atomic_with_scope() {
        let inst = DecodedInstruction {
            offset: 0,
            size: 4,
            operation: Operation::DeviceAtomic {
                op: AtomicOp::Add,
                rd: Some(3),
                addr: 4,
                value: 5,
                scope: Scope::Device,
            },
            predicate: 0,
            predicate_negated: false,
        };
        assert_eq!(format_instruction(&inst), "atomic_add.device r3, r4, r5");
    }

    #[test]
    fn test_format_mov_sr() {
        let inst = DecodedInstruction {
            offset: 0,
            size: 4,
            operation: Operation::MovSr { rd: 0, sr_index: 4 },
            predicate: 0,
            predicate_negated: false,
        };
        assert_eq!(format_instruction(&inst), "mov r0, sr_lane_id");
    }

    #[test]
    fn test_format_fence_release() {
        let inst = DecodedInstruction {
            offset: 0,
            size: 4,
            operation: Operation::FenceRelease { scope: Scope::Workgroup },
            predicate: 0,
            predicate_negated: false,
        };
        assert_eq!(format_instruction(&inst), "fence_release.workgroup");
    }
}
