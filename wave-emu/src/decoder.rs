// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Binary instruction decoder for WBIN format. Wraps wave-decode and provides
//!
//! a flat DecodedInstruction structure for the executor to dispatch on.

use crate::EmulatorError;

pub use wave_decode::{
    AtomicOp, BitOpType, CmpOp, ControlOp, CvtType, F16Op, F16PackedOp, F64DivSqrtOp, F64Op,
    FUnaryOp, MemWidth, MiscOp, Opcode, Scope, SyncOp, WaveOpType, WaveReduceType,
};

pub use wave_decode::opcodes::{
    MISC_OP_FLAG, SYNC_OP_FLAG,
};

/// Flat decoded instruction for executor dispatch
#[derive(Debug, Clone)]
pub struct DecodedInstruction {
    pub opcode: Opcode,
    pub rd: u8,
    pub rs1: u8,
    pub rs2: u8,
    pub rs3: u8,
    pub rs4: u8,
    pub modifier: u8,
    pub scope: u8,
    pub pred_reg: u8,
    pub pred_neg: bool,
    pub flags: u8,
    pub immediate: u32,
    pub size: u32,
}

impl DecodedInstruction {
    pub fn is_predicated(&self) -> bool {
        self.pred_reg != 0 || self.pred_neg
    }

    pub fn is_sync_op(&self) -> bool {
        self.opcode == Opcode::Control && (self.flags & SYNC_OP_FLAG) != 0
    }

    pub fn is_misc_op(&self) -> bool {
        self.opcode == Opcode::Control && (self.flags & MISC_OP_FLAG) != 0
    }

    pub fn is_non_returning_atomic(&self) -> bool {
        (self.opcode == Opcode::LocalAtomic || self.opcode == Opcode::DeviceAtomic)
            && self.rd == 0
    }

    pub fn is_wave_reduce(&self) -> bool {
        self.opcode == Opcode::WaveOp && self.modifier >= 8
    }
}

pub struct Decoder<'a> {
    code: &'a [u8],
}

impl<'a> Decoder<'a> {
    pub fn new(code: &'a [u8]) -> Self {
        Self { code }
    }

    pub fn decode_at(&self, pc: u32) -> Result<DecodedInstruction, EmulatorError> {
        let decoded = wave_decode::decode_at(self.code, pc).map_err(|e| {
            EmulatorError::InvalidInstruction {
                pc,
                message: e.to_string(),
            }
        })?;

        Ok(convert_instruction(&decoded))
    }

    pub fn disassemble(&self, inst: &DecodedInstruction) -> String {
        let pred = if inst.is_predicated() {
            let neg = if inst.pred_neg { "!" } else { "" };
            format!("@{neg}p{} ", inst.pred_reg)
        } else {
            String::new()
        };

        let mnemonic = self.get_mnemonic(inst);
        let operands = self.get_operands(inst);

        if operands.is_empty() {
            format!("{pred}{mnemonic}")
        } else {
            format!("{pred}{mnemonic} {operands}")
        }
    }

    fn get_mnemonic(&self, inst: &DecodedInstruction) -> &'static str {
        match inst.opcode {
            Opcode::Iadd => "iadd",
            Opcode::Isub => "isub",
            Opcode::Imul => "imul",
            Opcode::ImulHi => "imul_hi",
            Opcode::Imad => "imad",
            Opcode::Idiv => "idiv",
            Opcode::Imod => "imod",
            Opcode::Ineg => "ineg",
            Opcode::Iabs => "iabs",
            Opcode::Imin => "imin",
            Opcode::Imax => "imax",
            Opcode::Iclamp => "iclamp",
            Opcode::Fadd => "fadd",
            Opcode::Fsub => "fsub",
            Opcode::Fmul => "fmul",
            Opcode::Fma => "fma",
            Opcode::Fdiv => "fdiv",
            Opcode::Fneg => "fneg",
            Opcode::Fabs => "fabs",
            Opcode::Fmin => "fmin",
            Opcode::Fmax => "fmax",
            Opcode::Fclamp => "fclamp",
            Opcode::Fsqrt => "fsqrt",
            Opcode::FUnaryOps => FUnaryOp::from_u8(inst.modifier).map_or("f_unknown", |op| op.mnemonic()),
            Opcode::F16Ops => F16Op::from_u8(inst.modifier).map_or("h_unknown", |op| op.mnemonic()),
            Opcode::F16PackedOps => F16PackedOp::from_u8(inst.modifier).map_or("h2_unknown", |op| op.mnemonic()),
            Opcode::F64Ops => F64Op::from_u8(inst.modifier).map_or("d_unknown", |op| op.mnemonic()),
            Opcode::F64DivSqrt => F64DivSqrtOp::from_u8(inst.modifier).map_or("d_unknown", |op| op.mnemonic()),
            Opcode::And => "and",
            Opcode::Or => "or",
            Opcode::Xor => "xor",
            Opcode::Not => "not",
            Opcode::Shl => "shl",
            Opcode::Shr => "shr",
            Opcode::Sar => "sar",
            Opcode::BitOps => BitOpType::from_u8(inst.modifier).map_or("bit_unknown", |op| op.mnemonic()),
            Opcode::Icmp => {
                match inst.modifier {
                    0 => "icmp_eq",
                    1 => "icmp_ne",
                    2 => "icmp_lt",
                    3 => "icmp_le",
                    4 => "icmp_gt",
                    5 => "icmp_ge",
                    _ => "icmp_unknown",
                }
            }
            Opcode::Ucmp => {
                match inst.modifier {
                    2 => "ucmp_lt",
                    3 => "ucmp_le",
                    _ => "ucmp_unknown",
                }
            }
            Opcode::Fcmp => {
                match inst.modifier {
                    0 => "fcmp_eq",
                    1 => "fcmp_ne",
                    2 => "fcmp_lt",
                    3 => "fcmp_le",
                    4 => "fcmp_gt",
                    6 => "fcmp_ord",
                    7 => "fcmp_unord",
                    _ => "fcmp_unknown",
                }
            }
            Opcode::Select => "select",
            Opcode::Cvt => CvtType::from_u8(inst.modifier).map_or("cvt_unknown", |ct| ct.mnemonic()),
            Opcode::LocalLoad => {
                match inst.modifier {
                    0 => "local_load_u8",
                    1 => "local_load_u16",
                    2 => "local_load_u32",
                    3 => "local_load_u64",
                    _ => "local_load_unknown",
                }
            }
            Opcode::LocalStore => {
                match inst.modifier {
                    0 => "local_store_u8",
                    1 => "local_store_u16",
                    2 => "local_store_u32",
                    3 => "local_store_u64",
                    _ => "local_store_unknown",
                }
            }
            Opcode::DeviceLoad => {
                match inst.modifier {
                    0 => "device_load_u8",
                    1 => "device_load_u16",
                    2 => "device_load_u32",
                    3 => "device_load_u64",
                    4 => "device_load_u128",
                    _ => "device_load_unknown",
                }
            }
            Opcode::DeviceStore => {
                match inst.modifier {
                    0 => "device_store_u8",
                    1 => "device_store_u16",
                    2 => "device_store_u32",
                    3 => "device_store_u64",
                    4 => "device_store_u128",
                    _ => "device_store_unknown",
                }
            }
            Opcode::LocalAtomic => {
                match inst.modifier {
                    0 => "local_atomic_add",
                    1 => "local_atomic_sub",
                    2 => "local_atomic_min",
                    3 => "local_atomic_max",
                    4 => "local_atomic_and",
                    5 => "local_atomic_or",
                    6 => "local_atomic_xor",
                    7 => "local_atomic_exchange",
                    _ => "local_atomic_cas",
                }
            }
            Opcode::DeviceAtomic => {
                match inst.modifier {
                    0 => "atomic_add",
                    1 => "atomic_sub",
                    2 => "atomic_min",
                    3 => "atomic_max",
                    4 => "atomic_and",
                    5 => "atomic_or",
                    6 => "atomic_xor",
                    7 => "atomic_exchange",
                    _ => "atomic_cas",
                }
            }
            Opcode::WaveOp => {
                if inst.is_wave_reduce() {
                    WaveReduceType::from_u8(inst.modifier - 8).map_or("wave_reduce_unknown", |op| op.mnemonic())
                } else {
                    WaveOpType::from_u8(inst.modifier).map_or("wave_unknown", |op| op.mnemonic())
                }
            }
            Opcode::Control => {
                if inst.is_sync_op() {
                    SyncOp::from_u8(inst.modifier).map_or("sync_unknown", |op| op.mnemonic())
                } else if inst.is_misc_op() {
                    MiscOp::from_u8(inst.modifier).map_or("misc_unknown", |op| op.mnemonic())
                } else {
                    ControlOp::from_u8(inst.modifier).map_or("control_unknown", |op| op.mnemonic())
                }
            }
        }
    }

    fn get_operands(&self, inst: &DecodedInstruction) -> String {
        match inst.opcode {
            Opcode::Iadd | Opcode::Isub | Opcode::Imul | Opcode::ImulHi | Opcode::Idiv | Opcode::Imod
            | Opcode::Imin | Opcode::Imax | Opcode::Fadd | Opcode::Fsub | Opcode::Fmul | Opcode::Fdiv
            | Opcode::Fmin | Opcode::Fmax | Opcode::And | Opcode::Or | Opcode::Xor | Opcode::Shl
            | Opcode::Shr | Opcode::Sar => {
                format!("r{}, r{}, r{}", inst.rd, inst.rs1, inst.rs2)
            }
            Opcode::Imad | Opcode::Iclamp | Opcode::Fma | Opcode::Fclamp => {
                format!("r{}, r{}, r{}, r{}", inst.rd, inst.rs1, inst.rs2, inst.rs3)
            }
            Opcode::Ineg | Opcode::Iabs | Opcode::Fneg | Opcode::Fabs | Opcode::Fsqrt | Opcode::Not => {
                format!("r{}, r{}", inst.rd, inst.rs1)
            }
            Opcode::FUnaryOps => format!("r{}, r{}", inst.rd, inst.rs1),
            Opcode::Icmp | Opcode::Ucmp | Opcode::Fcmp => {
                format!("p{}, r{}, r{}", inst.rd, inst.rs1, inst.rs2)
            }
            Opcode::Select => {
                format!("r{}, p{}, r{}, r{}", inst.rd, inst.modifier, inst.rs1, inst.rs2)
            }
            Opcode::Cvt => format!("r{}, r{}", inst.rd, inst.rs1),
            Opcode::LocalLoad | Opcode::DeviceLoad => format!("r{}, r{}", inst.rd, inst.rs1),
            Opcode::LocalStore | Opcode::DeviceStore => format!("r{}, r{}", inst.rs1, inst.rs2),
            Opcode::LocalAtomic | Opcode::DeviceAtomic => {
                if inst.is_non_returning_atomic() {
                    format!("r{}, r{}", inst.rs1, inst.rs2)
                } else {
                    format!("r{}, r{}, r{}", inst.rd, inst.rs1, inst.rs2)
                }
            }
            Opcode::WaveOp => {
                let mnemonic = self.get_mnemonic(inst);
                if mnemonic == "wave_ballot" {
                    format!("r{}, p{}", inst.rd, inst.rs1)
                } else if mnemonic == "wave_any" || mnemonic == "wave_all" {
                    format!("p{}, p{}", inst.rd, inst.rs1)
                } else if mnemonic.starts_with("wave_reduce") || mnemonic == "wave_prefix_sum" {
                    format!("r{}, r{}", inst.rd, inst.rs1)
                } else {
                    format!("r{}, r{}, r{}", inst.rd, inst.rs1, inst.rs2)
                }
            }
            Opcode::Control => {
                if inst.is_misc_op() {
                    match inst.modifier {
                        0 => format!("r{}, r{}", inst.rd, inst.rs1),
                        1 => format!("r{}, 0x{:08x}", inst.rd, inst.immediate),
                        2 => format!("r{}, sr_{}", inst.rd, inst.rs1),
                        _ => String::new(),
                    }
                } else if inst.is_sync_op() {
                    String::new()
                } else {
                    match inst.modifier {
                        0 | 4 | 5 => format!("p{}", inst.rs1),
                        7 => format!("0x{:08x}", inst.immediate),
                        _ => String::new(),
                    }
                }
            }
            _ => String::new(),
        }
    }
}

fn convert_instruction(decoded: &wave_decode::DecodedInstruction) -> DecodedInstruction {
    use wave_decode::Operation;

    let (opcode, rd, rs1, rs2, rs3, rs4, modifier, scope, flags, immediate) = match &decoded.operation {
        Operation::Iadd { rd, rs1, rs2 } => (Opcode::Iadd, *rd, *rs1, *rs2, 0, 0, 0, 0, 0, 0),
        Operation::Isub { rd, rs1, rs2 } => (Opcode::Isub, *rd, *rs1, *rs2, 0, 0, 0, 0, 0, 0),
        Operation::Imul { rd, rs1, rs2 } => (Opcode::Imul, *rd, *rs1, *rs2, 0, 0, 0, 0, 0, 0),
        Operation::ImulHi { rd, rs1, rs2 } => (Opcode::ImulHi, *rd, *rs1, *rs2, 0, 0, 0, 0, 0, 0),
        Operation::Imad { rd, rs1, rs2, rs3 } => (Opcode::Imad, *rd, *rs1, *rs2, *rs3, 0, 0, 0, 0, 0),
        Operation::Idiv { rd, rs1, rs2 } => (Opcode::Idiv, *rd, *rs1, *rs2, 0, 0, 0, 0, 0, 0),
        Operation::Imod { rd, rs1, rs2 } => (Opcode::Imod, *rd, *rs1, *rs2, 0, 0, 0, 0, 0, 0),
        Operation::Ineg { rd, rs1 } => (Opcode::Ineg, *rd, *rs1, 0, 0, 0, 0, 0, 0, 0),
        Operation::Iabs { rd, rs1 } => (Opcode::Iabs, *rd, *rs1, 0, 0, 0, 0, 0, 0, 0),
        Operation::Imin { rd, rs1, rs2 } => (Opcode::Imin, *rd, *rs1, *rs2, 0, 0, 0, 0, 0, 0),
        Operation::Imax { rd, rs1, rs2 } => (Opcode::Imax, *rd, *rs1, *rs2, 0, 0, 0, 0, 0, 0),
        Operation::Iclamp { rd, rs1, rs2, rs3 } => (Opcode::Iclamp, *rd, *rs1, *rs2, *rs3, 0, 0, 0, 0, 0),

        Operation::Fadd { rd, rs1, rs2 } => (Opcode::Fadd, *rd, *rs1, *rs2, 0, 0, 0, 0, 0, 0),
        Operation::Fsub { rd, rs1, rs2 } => (Opcode::Fsub, *rd, *rs1, *rs2, 0, 0, 0, 0, 0, 0),
        Operation::Fmul { rd, rs1, rs2 } => (Opcode::Fmul, *rd, *rs1, *rs2, 0, 0, 0, 0, 0, 0),
        Operation::Fma { rd, rs1, rs2, rs3 } => (Opcode::Fma, *rd, *rs1, *rs2, *rs3, 0, 0, 0, 0, 0),
        Operation::Fdiv { rd, rs1, rs2 } => (Opcode::Fdiv, *rd, *rs1, *rs2, 0, 0, 0, 0, 0, 0),
        Operation::Fneg { rd, rs1 } => (Opcode::Fneg, *rd, *rs1, 0, 0, 0, 0, 0, 0, 0),
        Operation::Fabs { rd, rs1 } => (Opcode::Fabs, *rd, *rs1, 0, 0, 0, 0, 0, 0, 0),
        Operation::Fmin { rd, rs1, rs2 } => (Opcode::Fmin, *rd, *rs1, *rs2, 0, 0, 0, 0, 0, 0),
        Operation::Fmax { rd, rs1, rs2 } => (Opcode::Fmax, *rd, *rs1, *rs2, 0, 0, 0, 0, 0, 0),
        Operation::Fclamp { rd, rs1, rs2, rs3 } => (Opcode::Fclamp, *rd, *rs1, *rs2, *rs3, 0, 0, 0, 0, 0),
        Operation::Fsqrt { rd, rs1 } => (Opcode::Fsqrt, *rd, *rs1, 0, 0, 0, 0, 0, 0, 0),
        Operation::FUnary { op, rd, rs1 } => (Opcode::FUnaryOps, *rd, *rs1, 0, 0, 0, *op as u8, 0, 0, 0),

        Operation::F16 { op, rd, rs1, rs2, rs3 } => {
            (Opcode::F16Ops, *rd, *rs1, *rs2, rs3.unwrap_or(0), 0, *op as u8, 0, 0, 0)
        }
        Operation::F16Packed { op, rd, rs1, rs2, rs3 } => {
            (Opcode::F16PackedOps, *rd, *rs1, *rs2, rs3.unwrap_or(0), 0, *op as u8, 0, 0, 0)
        }

        Operation::F64 { op, rd, rs1, rs2, rs3 } => {
            (Opcode::F64Ops, *rd, *rs1, *rs2, rs3.unwrap_or(0), 0, *op as u8, 0, 0, 0)
        }
        Operation::F64DivSqrt { op, rd, rs1, rs2 } => {
            (Opcode::F64DivSqrt, *rd, *rs1, rs2.unwrap_or(0), 0, 0, *op as u8, 0, 0, 0)
        }

        Operation::And { rd, rs1, rs2 } => (Opcode::And, *rd, *rs1, *rs2, 0, 0, 0, 0, 0, 0),
        Operation::Or { rd, rs1, rs2 } => (Opcode::Or, *rd, *rs1, *rs2, 0, 0, 0, 0, 0, 0),
        Operation::Xor { rd, rs1, rs2 } => (Opcode::Xor, *rd, *rs1, *rs2, 0, 0, 0, 0, 0, 0),
        Operation::Not { rd, rs1 } => (Opcode::Not, *rd, *rs1, 0, 0, 0, 0, 0, 0, 0),
        Operation::Shl { rd, rs1, rs2 } => (Opcode::Shl, *rd, *rs1, *rs2, 0, 0, 0, 0, 0, 0),
        Operation::Shr { rd, rs1, rs2 } => (Opcode::Shr, *rd, *rs1, *rs2, 0, 0, 0, 0, 0, 0),
        Operation::Sar { rd, rs1, rs2 } => (Opcode::Sar, *rd, *rs1, *rs2, 0, 0, 0, 0, 0, 0),
        Operation::BitOp { op, rd, rs1, rs2, rs3, rs4 } => {
            (Opcode::BitOps, *rd, *rs1, rs2.unwrap_or(0), rs3.unwrap_or(0), rs4.unwrap_or(0), *op as u8, 0, 0, 0)
        }

        Operation::Icmp { op, pd, rs1, rs2 } => (Opcode::Icmp, *pd, *rs1, *rs2, 0, 0, *op as u8, 0, 0, 0),
        Operation::Ucmp { op, pd, rs1, rs2 } => (Opcode::Ucmp, *pd, *rs1, *rs2, 0, 0, *op as u8, 0, 0, 0),
        Operation::Fcmp { op, pd, rs1, rs2 } => (Opcode::Fcmp, *pd, *rs1, *rs2, 0, 0, *op as u8, 0, 0, 0),

        Operation::Select { rd, ps, rs1, rs2 } => (Opcode::Select, *rd, *rs1, *rs2, 0, 0, *ps, 0, 0, 0),
        Operation::Cvt { cvt_type, rd, rs1 } => (Opcode::Cvt, *rd, *rs1, 0, 0, 0, *cvt_type as u8, 0, 0, 0),

        Operation::LocalLoad { width, rd, addr } => (Opcode::LocalLoad, *rd, *addr, 0, 0, 0, *width as u8, 0, 0, 0),
        Operation::LocalStore { width, addr, value } => (Opcode::LocalStore, 0, *addr, *value, 0, 0, *width as u8, 0, 0, 0),

        Operation::DeviceLoad { width, rd, addr } => (Opcode::DeviceLoad, *rd, *addr, 0, 0, 0, *width as u8, 0, 0, 0),
        Operation::DeviceStore { width, addr, value } => (Opcode::DeviceStore, 0, *addr, *value, 0, 0, *width as u8, 0, 0, 0),

        Operation::LocalAtomic { op, rd, addr, value } => {
            let rd_val = rd.unwrap_or(0);
            (Opcode::LocalAtomic, rd_val, *addr, *value, 0, 0, *op as u8, 0, 0, 0)
        }
        Operation::LocalAtomicCas { rd, addr, expected, desired } => {
            let rd_val = rd.unwrap_or(0);
            (Opcode::LocalAtomic, rd_val, *addr, *expected, *desired, 0, 8, 0, 0, 0)
        }
        Operation::DeviceAtomic { op, rd, addr, value, scope } => {
            let rd_val = rd.unwrap_or(0);
            (Opcode::DeviceAtomic, rd_val, *addr, *value, 0, 0, *op as u8, *scope as u8, 0, 0)
        }
        Operation::DeviceAtomicCas { rd, addr, expected, desired, scope } => {
            let rd_val = rd.unwrap_or(0);
            (Opcode::DeviceAtomic, rd_val, *addr, *expected, *desired, 0, 8, *scope as u8, 0, 0)
        }

        Operation::WaveOp { op, rd, rs1, rs2 } => {
            (Opcode::WaveOp, *rd, *rs1, rs2.unwrap_or(0), 0, 0, *op as u8, 0, 0, 0)
        }
        Operation::WaveReduce { op, rd, rs1 } => {
            (Opcode::WaveOp, *rd, *rs1, 0, 0, 0, *op as u8 + 8, 0, 0, 0)
        }
        Operation::WaveBallot { rd, ps } => (Opcode::WaveOp, *rd, *ps, 0, 0, 0, WaveOpType::Ballot as u8, 0, 0, 0),
        Operation::WaveVote { op, pd, ps } => (Opcode::WaveOp, *pd, *ps, 0, 0, 0, *op as u8, 0, 0, 0),

        Operation::If { ps } => (Opcode::Control, 0, *ps, 0, 0, 0, ControlOp::If as u8, 0, 0, 0),
        Operation::Else => (Opcode::Control, 0, 0, 0, 0, 0, ControlOp::Else as u8, 0, 0, 0),
        Operation::Endif => (Opcode::Control, 0, 0, 0, 0, 0, ControlOp::Endif as u8, 0, 0, 0),
        Operation::Loop => (Opcode::Control, 0, 0, 0, 0, 0, ControlOp::Loop as u8, 0, 0, 0),
        Operation::Break { ps } => (Opcode::Control, 0, *ps, 0, 0, 0, ControlOp::Break as u8, 0, 0, 0),
        Operation::Continue { ps } => (Opcode::Control, 0, *ps, 0, 0, 0, ControlOp::Continue as u8, 0, 0, 0),
        Operation::Endloop => (Opcode::Control, 0, 0, 0, 0, 0, ControlOp::Endloop as u8, 0, 0, 0),
        Operation::Call { target } => (Opcode::Control, 0, 0, 0, 0, 0, ControlOp::Call as u8, 0, 0, *target),

        Operation::Return => (Opcode::Control, 0, 0, 0, 0, 0, SyncOp::Return as u8, 0, SYNC_OP_FLAG, 0),
        Operation::Halt => (Opcode::Control, 0, 0, 0, 0, 0, SyncOp::Halt as u8, 0, SYNC_OP_FLAG, 0),
        Operation::Barrier => (Opcode::Control, 0, 0, 0, 0, 0, SyncOp::Barrier as u8, 0, SYNC_OP_FLAG, 0),
        Operation::FenceAcquire { scope } => (Opcode::Control, 0, 0, 0, 0, 0, SyncOp::FenceAcquire as u8, *scope as u8, SYNC_OP_FLAG, 0),
        Operation::FenceRelease { scope } => (Opcode::Control, 0, 0, 0, 0, 0, SyncOp::FenceRelease as u8, *scope as u8, SYNC_OP_FLAG, 0),
        Operation::FenceAcqRel { scope } => (Opcode::Control, 0, 0, 0, 0, 0, SyncOp::FenceAcqRel as u8, *scope as u8, SYNC_OP_FLAG, 0),
        Operation::Wait => (Opcode::Control, 0, 0, 0, 0, 0, SyncOp::Wait as u8, 0, SYNC_OP_FLAG, 0),
        Operation::Nop => (Opcode::Control, 0, 0, 0, 0, 0, SyncOp::Nop as u8, 0, SYNC_OP_FLAG, 0),

        Operation::Mov { rd, rs1 } => (Opcode::Control, *rd, *rs1, 0, 0, 0, MiscOp::Mov as u8, 0, MISC_OP_FLAG, 0),
        Operation::MovImm { rd, imm } => (Opcode::Control, *rd, 0, 0, 0, 0, MiscOp::MovImm as u8, 0, MISC_OP_FLAG, *imm),
        Operation::MovSr { rd, sr_index } => (Opcode::Control, *rd, *sr_index, 0, 0, 0, MiscOp::MovSr as u8, 0, MISC_OP_FLAG, 0),

        Operation::Unknown { opcode, word0, word1: _ } => {
            let opc = Opcode::from_u8(*opcode).unwrap_or(Opcode::Control);
            (opc, 0, 0, 0, 0, 0, 0, 0, 0, *word0)
        }
    };

    DecodedInstruction {
        opcode,
        rd,
        rs1,
        rs2,
        rs3,
        rs4,
        modifier,
        scope,
        pred_reg: decoded.predicate,
        pred_neg: decoded.predicate_negated,
        flags,
        immediate,
        size: u32::from(decoded.size),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_base(opcode: u8, rd: u8, rs1: u8, rs2: u8, modifier: u8, flags: u8) -> Vec<u8> {
        let word = ((u32::from(opcode) & 0x3F) << 26)
            | ((u32::from(rd) & 0x1F) << 21)
            | ((u32::from(rs1) & 0x1F) << 16)
            | ((u32::from(rs2) & 0x1F) << 11)
            | ((u32::from(modifier) & 0x0F) << 7)
            | ((u32::from(flags) & 0x03));
        word.to_le_bytes().to_vec()
    }

    #[test]
    fn test_decoder_iadd() {
        let code = encode_base(0x00, 1, 2, 3, 0, 0);
        let decoder = Decoder::new(&code);
        let inst = decoder.decode_at(0).unwrap();

        assert_eq!(inst.opcode, Opcode::Iadd);
        assert_eq!(inst.rd, 1);
        assert_eq!(inst.rs1, 2);
        assert_eq!(inst.rs2, 3);
        assert_eq!(inst.size, 4);
    }

    #[test]
    fn test_decoder_halt() {
        let code = encode_base(0x3F, 0, 0, 0, 1, SYNC_OP_FLAG);
        let decoder = Decoder::new(&code);
        let inst = decoder.decode_at(0).unwrap();

        assert_eq!(inst.opcode, Opcode::Control);
        assert!(inst.is_sync_op());
        assert_eq!(inst.modifier, 1);
    }

    #[test]
    fn test_decoder_mov_imm() {
        let word0 = encode_base(0x3F, 5, 0, 0, 1, MISC_OP_FLAG);
        let word1 = 0x12345678u32;

        let mut code = word0;
        code.extend_from_slice(&word1.to_le_bytes());

        let decoder = Decoder::new(&code);
        let inst = decoder.decode_at(0).unwrap();

        assert_eq!(inst.opcode, Opcode::Control);
        assert!(inst.is_misc_op());
        assert_eq!(inst.rd, 5);
        assert_eq!(inst.immediate, 0x12345678);
        assert_eq!(inst.size, 8);
    }

    #[test]
    fn test_decoder_disassemble() {
        let code = encode_base(0x00, 1, 2, 3, 0, 0);
        let decoder = Decoder::new(&code);
        let inst = decoder.decode_at(0).unwrap();
        let disasm = decoder.disassemble(&inst);

        assert_eq!(disasm, "iadd r1, r2, r3");
    }

    #[test]
    fn test_decoder_out_of_bounds() {
        let code = vec![0u8; 2];
        let decoder = Decoder::new(&code);
        assert!(decoder.decode_at(0).is_err());
    }
}
