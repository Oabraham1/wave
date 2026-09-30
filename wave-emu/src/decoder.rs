// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Binary instruction decoder for WBIN format. Wraps wave-decode and provides a flat
//! `DecodedInstruction` structure for the executor to dispatch on.

use crate::EmulatorError;

pub use wave_decode::{
    AtomicOp, Bf16Op, Bf16PackedOp, BitOpType, CmpOp, ControlOp, CvtType, F16Op, F16PackedOp,
    F64DivSqrtOp, F64Op, FUnaryOp, MemWidth, MiscOp, MmaOp, MmaPrecision, Opcode, Scope, SyncOp,
    WaveOpType, WaveReduceType,
};

pub use wave_decode::opcodes::SYNC_MODIFIER_OFFSET;

/// Flat decoded instruction for executor dispatch
#[derive(Debug, Clone, Copy)]
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
    pub immediate: u32,
    pub size: u32,
}

impl DecodedInstruction {
    #[must_use]
    pub fn is_predicated(&self) -> bool {
        self.pred_reg != 0 || self.pred_neg
    }

    #[must_use]
    pub fn is_sync_op(&self) -> bool {
        self.opcode == Opcode::Control && self.modifier >= SYNC_MODIFIER_OFFSET
    }

    #[must_use]
    pub fn is_misc_op(&self) -> bool {
        self.opcode == Opcode::Misc
    }

    #[must_use]
    pub fn is_non_returning_atomic(&self) -> bool {
        (self.opcode == Opcode::LocalAtomic || self.opcode == Opcode::DeviceAtomic) && self.rd == 0
    }

    #[must_use]
    pub fn is_wave_reduce(&self) -> bool {
        self.opcode == Opcode::WaveOp && self.modifier >= 8
    }
}

#[derive(Debug)]
pub struct Decoder<'a> {
    code: &'a [u8],
}

impl<'a> Decoder<'a> {
    #[must_use]
    pub fn new(code: &'a [u8]) -> Self {
        Self { code }
    }

    /// Decode the instruction at a byte offset.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid offset, truncated instruction, or unknown encoding.
    pub fn decode_at(&self, pc: u32) -> Result<DecodedInstruction, EmulatorError> {
        let decoded = wave_decode::decode_at(self.code, pc).map_err(|e| {
            EmulatorError::InvalidInstruction {
                pc,
                message: e.to_string(),
            }
        })?;

        Ok(convert_instruction(&decoded))
    }

    #[must_use]
    pub fn disassemble(&self, inst: &DecodedInstruction) -> String {
        let pred = if inst.is_predicated() {
            let neg = if inst.pred_neg { "!" } else { "" };
            format!("@{neg}p{} ", inst.pred_reg)
        } else {
            String::new()
        };

        let mnemonic = Self::get_mnemonic(inst);
        let operands = Self::get_operands(inst);

        if operands.is_empty() {
            format!("{pred}{mnemonic}")
        } else {
            format!("{pred}{mnemonic} {operands}")
        }
    }

    // The opcode table maps each encoding to its mnemonic.

    fn get_mnemonic(inst: &DecodedInstruction) -> &'static str {
        match inst.opcode {
            Opcode::Iadd
            | Opcode::Isub
            | Opcode::Imul
            | Opcode::ImulHi
            | Opcode::Imad
            | Opcode::Idiv
            | Opcode::Imod
            | Opcode::Ineg
            | Opcode::Iabs
            | Opcode::Imin
            | Opcode::Imax
            | Opcode::Iclamp => Self::get_mnemonic_integer(inst),
            Opcode::Fadd
            | Opcode::Fsub
            | Opcode::Fmul
            | Opcode::Fma
            | Opcode::Fdiv
            | Opcode::Fneg
            | Opcode::Fabs
            | Opcode::Fmin
            | Opcode::Fmax
            | Opcode::Fclamp
            | Opcode::Fsqrt
            | Opcode::FUnaryOps => Self::get_mnemonic_float(inst),
            Opcode::F16Ops | Opcode::F16PackedOps | Opcode::Bf16Ops | Opcode::Bf16PackedOps => {
                Self::get_mnemonic_half(inst)
            }
            Opcode::F64Ops | Opcode::F64DivSqrt => Self::get_mnemonic_double(inst),
            Opcode::And
            | Opcode::Or
            | Opcode::Xor
            | Opcode::Not
            | Opcode::Shl
            | Opcode::Shr
            | Opcode::Sar
            | Opcode::BitOps => Self::get_mnemonic_bitwise(inst),
            Opcode::Icmp | Opcode::Ucmp | Opcode::Fcmp | Opcode::Select | Opcode::Cvt => {
                Self::get_mnemonic_compare(inst)
            }
            Opcode::LocalLoad | Opcode::LocalStore | Opcode::DeviceLoad | Opcode::DeviceStore => {
                Self::get_mnemonic_load_store(inst)
            }
            Opcode::LocalAtomic | Opcode::DeviceAtomic => Self::get_mnemonic_atomic(inst),
            Opcode::WaveOp | Opcode::Mma => Self::get_mnemonic_wave(inst),
            Opcode::Control => Self::get_mnemonic_control(inst),
            Opcode::Misc => Self::get_mnemonic_move(inst),
        }
    }

    fn get_mnemonic_integer(inst: &DecodedInstruction) -> &'static str {
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
            _ => unreachable!("get_mnemonic routes only integer opcodes here"),
        }
    }

    fn get_mnemonic_float(inst: &DecodedInstruction) -> &'static str {
        match inst.opcode {
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
            Opcode::FUnaryOps => {
                FUnaryOp::from_u8(inst.modifier).map_or("f_unknown", |op| op.mnemonic())
            }
            _ => unreachable!("get_mnemonic routes only float opcodes here"),
        }
    }

    fn get_mnemonic_half(inst: &DecodedInstruction) -> &'static str {
        match inst.opcode {
            Opcode::F16Ops => F16Op::from_u8(inst.modifier).map_or("h_unknown", |op| op.mnemonic()),
            Opcode::F16PackedOps => {
                F16PackedOp::from_u8(inst.modifier).map_or("h2_unknown", |op| op.mnemonic())
            }
            Opcode::Bf16Ops => {
                Bf16Op::from_u8(inst.modifier).map_or("b_unknown", |op| op.mnemonic())
            }
            Opcode::Bf16PackedOps => {
                Bf16PackedOp::from_u8(inst.modifier).map_or("b2_unknown", |op| op.mnemonic())
            }
            _ => unreachable!("get_mnemonic routes only half opcodes here"),
        }
    }

    fn get_mnemonic_double(inst: &DecodedInstruction) -> &'static str {
        match inst.opcode {
            Opcode::F64Ops => F64Op::from_u8(inst.modifier).map_or("d_unknown", |op| op.mnemonic()),
            Opcode::F64DivSqrt => {
                F64DivSqrtOp::from_u8(inst.modifier).map_or("d_unknown", |op| op.mnemonic())
            }
            _ => unreachable!("get_mnemonic routes only double opcodes here"),
        }
    }

    fn get_mnemonic_bitwise(inst: &DecodedInstruction) -> &'static str {
        match inst.opcode {
            Opcode::And => "and",
            Opcode::Or => "or",
            Opcode::Xor => "xor",
            Opcode::Not => "not",
            Opcode::Shl => "shl",
            Opcode::Shr => "shr",
            Opcode::Sar => "sar",
            Opcode::BitOps => {
                BitOpType::from_u8(inst.modifier).map_or("bit_unknown", |op| op.mnemonic())
            }
            _ => unreachable!("get_mnemonic routes only bitwise opcodes here"),
        }
    }

    fn get_mnemonic_compare(inst: &DecodedInstruction) -> &'static str {
        match inst.opcode {
            Opcode::Icmp => match inst.modifier {
                0 => "icmp_eq",
                1 => "icmp_ne",
                2 => "icmp_lt",
                3 => "icmp_le",
                4 => "icmp_gt",
                5 => "icmp_ge",
                _ => "icmp_unknown",
            },
            Opcode::Ucmp => match inst.modifier {
                2 => "ucmp_lt",
                3 => "ucmp_le",
                _ => "ucmp_unknown",
            },
            Opcode::Fcmp => match inst.modifier {
                0 => "fcmp_eq",
                1 => "fcmp_ne",
                2 => "fcmp_lt",
                3 => "fcmp_le",
                4 => "fcmp_gt",
                6 => "fcmp_ord",
                7 => "fcmp_unord",
                _ => "fcmp_unknown",
            },
            Opcode::Select => "select",
            Opcode::Cvt => {
                CvtType::from_u8(inst.modifier).map_or("cvt_unknown", |ct| ct.mnemonic())
            }
            _ => unreachable!("get_mnemonic routes only compare opcodes here"),
        }
    }

    fn get_mnemonic_load_store(inst: &DecodedInstruction) -> &'static str {
        match inst.opcode {
            Opcode::LocalLoad => match inst.modifier {
                0 => "local_load_u8",
                1 => "local_load_u16",
                2 => "local_load_u32",
                3 => "local_load_u64",
                _ => "local_load_unknown",
            },
            Opcode::LocalStore => match inst.modifier {
                0 => "local_store_u8",
                1 => "local_store_u16",
                2 => "local_store_u32",
                3 => "local_store_u64",
                _ => "local_store_unknown",
            },
            Opcode::DeviceLoad => match inst.modifier {
                0 => "device_load_u8",
                1 => "device_load_u16",
                2 => "device_load_u32",
                3 => "device_load_u64",
                4 => "device_load_u128",
                _ => "device_load_unknown",
            },
            Opcode::DeviceStore => match inst.modifier {
                0 => "device_store_u8",
                1 => "device_store_u16",
                2 => "device_store_u32",
                3 => "device_store_u64",
                4 => "device_store_u128",
                _ => "device_store_unknown",
            },
            _ => unreachable!("get_mnemonic routes only load_store opcodes here"),
        }
    }

    fn get_mnemonic_atomic(inst: &DecodedInstruction) -> &'static str {
        match inst.opcode {
            Opcode::LocalAtomic => match inst.modifier {
                0 => "local_atomic_add",
                1 => "local_atomic_sub",
                2 => "local_atomic_min",
                3 => "local_atomic_max",
                4 => "local_atomic_and",
                5 => "local_atomic_or",
                6 => "local_atomic_xor",
                7 => "local_atomic_exchange",
                _ => "local_atomic_cas",
            },
            Opcode::DeviceAtomic => match inst.modifier {
                0 => "atomic_add",
                1 => "atomic_sub",
                2 => "atomic_min",
                3 => "atomic_max",
                4 => "atomic_and",
                5 => "atomic_or",
                6 => "atomic_xor",
                7 => "atomic_exchange",
                _ => "atomic_cas",
            },
            _ => unreachable!("get_mnemonic routes only atomic opcodes here"),
        }
    }

    fn get_mnemonic_wave(inst: &DecodedInstruction) -> &'static str {
        match inst.opcode {
            Opcode::WaveOp => {
                if inst.is_wave_reduce() {
                    WaveReduceType::from_u8(inst.modifier - 8)
                        .map_or("wave_reduce_unknown", |op| op.mnemonic())
                } else {
                    WaveOpType::from_u8(inst.modifier).map_or("wave_unknown", |op| op.mnemonic())
                }
            }
            Opcode::Mma => match MmaOp::from_u8(inst.modifier) {
                Some(op) => op.mnemonic(),
                None => "mma_unknown",
            },
            _ => unreachable!("get_mnemonic routes only wave opcodes here"),
        }
    }

    fn get_mnemonic_control(inst: &DecodedInstruction) -> &'static str {
        match inst.opcode {
            Opcode::Control => {
                if inst.is_sync_op() {
                    SyncOp::from_u8(inst.modifier - SYNC_MODIFIER_OFFSET)
                        .map_or("sync_unknown", |op| op.mnemonic())
                } else {
                    ControlOp::from_u8(inst.modifier).map_or("control_unknown", |op| op.mnemonic())
                }
            }
            _ => unreachable!("get_mnemonic routes only control opcodes here"),
        }
    }

    fn get_mnemonic_move(inst: &DecodedInstruction) -> &'static str {
        match inst.opcode {
            Opcode::Misc => {
                MiscOp::from_u8(inst.modifier).map_or("misc_unknown", |op| op.mnemonic())
            }
            _ => unreachable!("get_mnemonic routes only move opcodes here"),
        }
    }

    fn get_operands(inst: &DecodedInstruction) -> String {
        match inst.opcode {
            Opcode::Iadd
            | Opcode::Isub
            | Opcode::Imul
            | Opcode::ImulHi
            | Opcode::Idiv
            | Opcode::Imod
            | Opcode::Imin
            | Opcode::Imax
            | Opcode::Fadd
            | Opcode::Fsub
            | Opcode::Fmul
            | Opcode::Fdiv
            | Opcode::Fmin
            | Opcode::Fmax
            | Opcode::And
            | Opcode::Or
            | Opcode::Xor
            | Opcode::Shl
            | Opcode::Shr
            | Opcode::Sar
            | Opcode::F16Ops
            | Opcode::F16PackedOps
            | Opcode::Bf16Ops
            | Opcode::Bf16PackedOps
            | Opcode::F64Ops
            | Opcode::F64DivSqrt => {
                if inst.rs3 != 0 {
                    format!("r{}, r{}, r{}, r{}", inst.rd, inst.rs1, inst.rs2, inst.rs3)
                } else {
                    format!("r{}, r{}, r{}", inst.rd, inst.rs1, inst.rs2)
                }
            }
            Opcode::Imad | Opcode::Iclamp | Opcode::Fma | Opcode::Fclamp => {
                format!("r{}, r{}, r{}, r{}", inst.rd, inst.rs1, inst.rs2, inst.rs3)
            }
            Opcode::Ineg
            | Opcode::Iabs
            | Opcode::Fneg
            | Opcode::Fabs
            | Opcode::Fsqrt
            | Opcode::Not
            | Opcode::FUnaryOps
            | Opcode::Cvt
            | Opcode::LocalLoad
            | Opcode::DeviceLoad => {
                format!("r{}, r{}", inst.rd, inst.rs1)
            }
            Opcode::Icmp | Opcode::Ucmp | Opcode::Fcmp => {
                format!("p{}, r{}, r{}", inst.rd, inst.rs1, inst.rs2)
            }
            Opcode::Select => {
                format!(
                    "r{}, p{}, r{}, r{}",
                    inst.rd, inst.modifier, inst.rs1, inst.rs2
                )
            }
            Opcode::LocalStore | Opcode::DeviceStore => format!("r{}, r{}", inst.rs1, inst.rs2),
            Opcode::LocalAtomic | Opcode::DeviceAtomic => {
                if inst.is_non_returning_atomic() {
                    format!("r{}, r{}", inst.rs1, inst.rs2)
                } else {
                    format!("r{}, r{}, r{}", inst.rd, inst.rs1, inst.rs2)
                }
            }
            Opcode::WaveOp => {
                let mnemonic = Self::get_mnemonic(inst);
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
                if inst.is_sync_op() {
                    String::new()
                } else {
                    match inst.modifier {
                        0 | 4 | 5 => format!("p{}", inst.rs1),
                        7 => format!("0x{:08x}", inst.immediate),
                        _ => String::new(),
                    }
                }
            }
            Opcode::Misc => match inst.modifier {
                0 => format!("r{}, r{}", inst.rd, inst.rs1),
                1 => format!("r{}, 0x{:08x}", inst.rd, inst.immediate),
                2 => format!("r{}, sr_{}", inst.rd, inst.rs1),
                _ => String::new(),
            },
            Opcode::Mma => format!("r{}, r{}, r{}, r{}", inst.rd, inst.rs1, inst.rs2, inst.rs3),
            Opcode::BitOps => String::new(),
        }
    }
}

// The match translates every shared decoder operation.

fn convert_instruction(decoded: &wave_decode::DecodedInstruction) -> DecodedInstruction {
    use wave_decode::Operation;
    let mut inst = empty_instruction(decoded);
    match &decoded.operation {
        Operation::Iadd { .. }
        | Operation::Isub { .. }
        | Operation::Imul { .. }
        | Operation::ImulHi { .. }
        | Operation::Imad { .. }
        | Operation::Idiv { .. }
        | Operation::Imod { .. }
        | Operation::Ineg { .. }
        | Operation::Iabs { .. }
        | Operation::Imin { .. }
        | Operation::Imax { .. }
        | Operation::Iclamp { .. } => convert_integer(&decoded.operation, &mut inst),
        Operation::Fadd { .. }
        | Operation::Fsub { .. }
        | Operation::Fmul { .. }
        | Operation::Fma { .. }
        | Operation::Fdiv { .. }
        | Operation::Fneg { .. }
        | Operation::Fabs { .. }
        | Operation::Fmin { .. }
        | Operation::Fmax { .. }
        | Operation::Fclamp { .. }
        | Operation::Fsqrt { .. }
        | Operation::FUnary { .. } => convert_float(&decoded.operation, &mut inst),
        Operation::F16 { .. }
        | Operation::F16Packed { .. }
        | Operation::Bf16 { .. }
        | Operation::Bf16Packed { .. } => convert_half(&decoded.operation, &mut inst),
        Operation::F64 { .. } | Operation::F64DivSqrt { .. } => {
            convert_double(&decoded.operation, &mut inst);
        }
        Operation::And { .. }
        | Operation::Or { .. }
        | Operation::Xor { .. }
        | Operation::Not { .. }
        | Operation::Shl { .. }
        | Operation::Shr { .. }
        | Operation::Sar { .. }
        | Operation::BitOp { .. } => convert_bitwise(&decoded.operation, &mut inst),
        Operation::Icmp { .. }
        | Operation::Ucmp { .. }
        | Operation::Fcmp { .. }
        | Operation::Select { .. }
        | Operation::Cvt { .. } => convert_compare(&decoded.operation, &mut inst),
        Operation::LocalLoad { .. }
        | Operation::LocalStore { .. }
        | Operation::DeviceLoad { .. }
        | Operation::DeviceStore { .. } => convert_load_store(&decoded.operation, &mut inst),
        Operation::LocalAtomic { .. }
        | Operation::LocalAtomicCas { .. }
        | Operation::DeviceAtomic { .. }
        | Operation::DeviceAtomicCas { .. } => convert_atomic(&decoded.operation, &mut inst),
        Operation::WaveOp { .. }
        | Operation::WaveReduce { .. }
        | Operation::WaveBallot { .. }
        | Operation::WaveVote { .. }
        | Operation::MmaLoadA { .. }
        | Operation::MmaLoadB { .. }
        | Operation::MmaStoreC { .. }
        | Operation::MmaCompute { .. } => convert_wave(&decoded.operation, &mut inst),
        Operation::If { .. }
        | Operation::Else
        | Operation::Endif
        | Operation::Loop
        | Operation::Break { .. }
        | Operation::Continue { .. }
        | Operation::Endloop
        | Operation::Call { .. }
        | Operation::Return
        | Operation::Halt
        | Operation::Barrier
        | Operation::FenceAcquire { .. }
        | Operation::FenceRelease { .. }
        | Operation::FenceAcqRel { .. }
        | Operation::Wait
        | Operation::Nop => convert_control(&decoded.operation, &mut inst),
        Operation::Mov { .. }
        | Operation::MovImm { .. }
        | Operation::MovSr { .. }
        | Operation::Unknown { .. } => convert_move(&decoded.operation, &mut inst),
    }
    inst
}

fn empty_instruction(decoded: &wave_decode::DecodedInstruction) -> DecodedInstruction {
    DecodedInstruction {
        opcode: Opcode::Misc,
        rd: 0,
        rs1: 0,
        rs2: 0,
        rs3: 0,
        rs4: 0,
        modifier: 0,
        scope: 0,
        immediate: 0,
        pred_reg: decoded.predicate,
        pred_neg: decoded.predicate_negated,
        size: u32::from(decoded.size),
    }
}

fn convert_integer(op: &wave_decode::Operation, inst: &mut DecodedInstruction) {
    use wave_decode::Operation;
    match op {
        Operation::Iadd { rd, rs1, rs2 } => {
            inst.opcode = Opcode::Iadd;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
        }
        Operation::Isub { rd, rs1, rs2 } => {
            inst.opcode = Opcode::Isub;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
        }
        Operation::Imul { rd, rs1, rs2 } => {
            inst.opcode = Opcode::Imul;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
        }
        Operation::ImulHi { rd, rs1, rs2 } => {
            inst.opcode = Opcode::ImulHi;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
        }
        Operation::Imad { rd, rs1, rs2, rs3 } => {
            inst.opcode = Opcode::Imad;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
            inst.rs3 = *rs3;
        }
        Operation::Idiv { rd, rs1, rs2 } => {
            inst.opcode = Opcode::Idiv;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
        }
        Operation::Imod { rd, rs1, rs2 } => {
            inst.opcode = Opcode::Imod;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
        }
        Operation::Ineg { rd, rs1 } => {
            inst.opcode = Opcode::Ineg;
            inst.rd = *rd;
            inst.rs1 = *rs1;
        }
        Operation::Iabs { rd, rs1 } => {
            inst.opcode = Opcode::Iabs;
            inst.rd = *rd;
            inst.rs1 = *rs1;
        }
        Operation::Imin { rd, rs1, rs2 } => {
            inst.opcode = Opcode::Imin;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
        }
        Operation::Imax { rd, rs1, rs2 } => {
            inst.opcode = Opcode::Imax;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
        }
        Operation::Iclamp { rd, rs1, rs2, rs3 } => {
            inst.opcode = Opcode::Iclamp;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
            inst.rs3 = *rs3;
        }
        _ => unreachable!("convert_instruction routes only integer operations here"),
    }
}

fn convert_float(op: &wave_decode::Operation, inst: &mut DecodedInstruction) {
    use wave_decode::Operation;
    match op {
        Operation::Fadd { rd, rs1, rs2 } => {
            inst.opcode = Opcode::Fadd;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
        }
        Operation::Fsub { rd, rs1, rs2 } => {
            inst.opcode = Opcode::Fsub;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
        }
        Operation::Fmul { rd, rs1, rs2 } => {
            inst.opcode = Opcode::Fmul;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
        }
        Operation::Fma { rd, rs1, rs2, rs3 } => {
            inst.opcode = Opcode::Fma;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
            inst.rs3 = *rs3;
        }
        Operation::Fdiv { rd, rs1, rs2 } => {
            inst.opcode = Opcode::Fdiv;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
        }
        Operation::Fneg { rd, rs1 } => {
            inst.opcode = Opcode::Fneg;
            inst.rd = *rd;
            inst.rs1 = *rs1;
        }
        Operation::Fabs { rd, rs1 } => {
            inst.opcode = Opcode::Fabs;
            inst.rd = *rd;
            inst.rs1 = *rs1;
        }
        Operation::Fmin { rd, rs1, rs2 } => {
            inst.opcode = Opcode::Fmin;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
        }
        Operation::Fmax { rd, rs1, rs2 } => {
            inst.opcode = Opcode::Fmax;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
        }
        Operation::Fclamp { rd, rs1, rs2, rs3 } => {
            inst.opcode = Opcode::Fclamp;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
            inst.rs3 = *rs3;
        }
        Operation::Fsqrt { rd, rs1 } => {
            inst.opcode = Opcode::Fsqrt;
            inst.rd = *rd;
            inst.rs1 = *rs1;
        }
        Operation::FUnary { op, rd, rs1 } => {
            inst.opcode = Opcode::FUnaryOps;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.modifier = *op as u8;
        }
        _ => unreachable!("convert_instruction routes only float operations here"),
    }
}

fn convert_half(op: &wave_decode::Operation, inst: &mut DecodedInstruction) {
    use wave_decode::Operation;
    match op {
        Operation::F16 {
            op,
            rd,
            rs1,
            rs2,
            rs3,
        } => {
            inst.opcode = Opcode::F16Ops;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
            inst.rs3 = rs3.unwrap_or(0);
            inst.modifier = *op as u8;
        }
        Operation::F16Packed {
            op,
            rd,
            rs1,
            rs2,
            rs3,
        } => {
            inst.opcode = Opcode::F16PackedOps;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
            inst.rs3 = rs3.unwrap_or(0);
            inst.modifier = *op as u8;
        }
        Operation::Bf16 {
            op,
            rd,
            rs1,
            rs2,
            rs3,
        } => {
            inst.opcode = Opcode::Bf16Ops;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
            inst.rs3 = rs3.unwrap_or(0);
            inst.modifier = *op as u8;
        }
        Operation::Bf16Packed {
            op,
            rd,
            rs1,
            rs2,
            rs3,
        } => {
            inst.opcode = Opcode::Bf16PackedOps;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
            inst.rs3 = rs3.unwrap_or(0);
            inst.modifier = *op as u8;
        }
        _ => unreachable!("convert_instruction routes only half operations here"),
    }
}

fn convert_double(op: &wave_decode::Operation, inst: &mut DecodedInstruction) {
    use wave_decode::Operation;
    match op {
        Operation::F64 {
            op,
            rd,
            rs1,
            rs2,
            rs3,
        } => {
            inst.opcode = Opcode::F64Ops;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
            inst.rs3 = rs3.unwrap_or(0);
            inst.modifier = *op as u8;
        }
        Operation::F64DivSqrt { op, rd, rs1, rs2 } => {
            inst.opcode = Opcode::F64DivSqrt;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = rs2.unwrap_or(0);
            inst.modifier = *op as u8;
        }
        _ => unreachable!("convert_instruction routes only double operations here"),
    }
}

fn convert_bitwise(op: &wave_decode::Operation, inst: &mut DecodedInstruction) {
    use wave_decode::Operation;
    match op {
        Operation::And { rd, rs1, rs2 } => {
            inst.opcode = Opcode::And;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
        }
        Operation::Or { rd, rs1, rs2 } => {
            inst.opcode = Opcode::Or;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
        }
        Operation::Xor { rd, rs1, rs2 } => {
            inst.opcode = Opcode::Xor;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
        }
        Operation::Not { rd, rs1 } => {
            inst.opcode = Opcode::Not;
            inst.rd = *rd;
            inst.rs1 = *rs1;
        }
        Operation::Shl { rd, rs1, rs2 } => {
            inst.opcode = Opcode::Shl;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
        }
        Operation::Shr { rd, rs1, rs2 } => {
            inst.opcode = Opcode::Shr;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
        }
        Operation::Sar { rd, rs1, rs2 } => {
            inst.opcode = Opcode::Sar;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
        }
        Operation::BitOp {
            op,
            rd,
            rs1,
            rs2,
            rs3,
            rs4,
        } => {
            inst.opcode = Opcode::BitOps;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = rs2.unwrap_or(0);
            inst.rs3 = rs3.unwrap_or(0);
            inst.rs4 = rs4.unwrap_or(0);
            inst.modifier = *op as u8;
        }
        _ => unreachable!("convert_instruction routes only bitwise operations here"),
    }
}

fn convert_compare(op: &wave_decode::Operation, inst: &mut DecodedInstruction) {
    use wave_decode::Operation;
    match op {
        Operation::Icmp { op, pd, rs1, rs2 } => {
            inst.opcode = Opcode::Icmp;
            inst.rd = *pd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
            inst.modifier = *op as u8;
        }
        Operation::Ucmp { op, pd, rs1, rs2 } => {
            inst.opcode = Opcode::Ucmp;
            inst.rd = *pd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
            inst.modifier = *op as u8;
        }
        Operation::Fcmp { op, pd, rs1, rs2 } => {
            inst.opcode = Opcode::Fcmp;
            inst.rd = *pd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
            inst.modifier = *op as u8;
        }
        Operation::Select { rd, ps, rs1, rs2 } => {
            inst.opcode = Opcode::Select;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
            inst.modifier = *ps;
        }
        Operation::Cvt { cvt_type, rd, rs1 } => {
            inst.opcode = Opcode::Cvt;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.modifier = *cvt_type as u8;
        }
        _ => unreachable!("convert_instruction routes only compare operations here"),
    }
}

fn convert_load_store(op: &wave_decode::Operation, inst: &mut DecodedInstruction) {
    use wave_decode::Operation;
    match op {
        Operation::LocalLoad { width, rd, addr } => {
            inst.opcode = Opcode::LocalLoad;
            inst.rd = *rd;
            inst.rs1 = *addr;
            inst.modifier = *width as u8;
        }
        Operation::LocalStore { width, addr, value } => {
            inst.opcode = Opcode::LocalStore;
            inst.rs1 = *addr;
            inst.rs2 = *value;
            inst.modifier = *width as u8;
        }
        Operation::DeviceLoad { width, rd, addr } => {
            inst.opcode = Opcode::DeviceLoad;
            inst.rd = *rd;
            inst.rs1 = *addr;
            inst.modifier = *width as u8;
        }
        Operation::DeviceStore { width, addr, value } => {
            inst.opcode = Opcode::DeviceStore;
            inst.rs1 = *addr;
            inst.rs2 = *value;
            inst.modifier = *width as u8;
        }
        _ => unreachable!("convert_instruction routes only load_store operations here"),
    }
}

fn convert_atomic(op: &wave_decode::Operation, inst: &mut DecodedInstruction) {
    use wave_decode::Operation;
    match op {
        Operation::LocalAtomic {
            op,
            rd,
            addr,
            value,
        } => {
            let rd_val = rd.unwrap_or(0);
            inst.opcode = Opcode::LocalAtomic;
            inst.rd = rd_val;
            inst.rs1 = *addr;
            inst.rs2 = *value;
            inst.modifier = *op as u8;
        }
        Operation::LocalAtomicCas {
            rd,
            addr,
            expected,
            desired,
        } => {
            let rd_val = rd.unwrap_or(0);
            inst.opcode = Opcode::LocalAtomic;
            inst.rd = rd_val;
            inst.rs1 = *addr;
            inst.rs2 = *expected;
            inst.rs3 = *desired;
            inst.modifier = 8;
        }
        Operation::DeviceAtomic {
            op,
            rd,
            addr,
            value,
            scope,
        } => {
            let rd_val = rd.unwrap_or(0);
            inst.opcode = Opcode::DeviceAtomic;
            inst.rd = rd_val;
            inst.rs1 = *addr;
            inst.rs2 = *value;
            inst.modifier = *op as u8;
            inst.scope = *scope as u8;
        }
        Operation::DeviceAtomicCas {
            rd,
            addr,
            expected,
            desired,
            scope,
        } => {
            let rd_val = rd.unwrap_or(0);
            inst.opcode = Opcode::DeviceAtomic;
            inst.rd = rd_val;
            inst.rs1 = *addr;
            inst.rs2 = *expected;
            inst.rs3 = *desired;
            inst.modifier = 8;
            inst.scope = *scope as u8;
        }
        _ => unreachable!("convert_instruction routes only atomic operations here"),
    }
}

fn convert_wave(op: &wave_decode::Operation, inst: &mut DecodedInstruction) {
    use wave_decode::Operation;
    match op {
        Operation::WaveOp { op, rd, rs1, rs2 } => {
            inst.opcode = Opcode::WaveOp;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = rs2.unwrap_or(0);
            inst.modifier = *op as u8;
        }
        Operation::WaveReduce { op, rd, rs1 } => {
            inst.opcode = Opcode::WaveOp;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.modifier = *op as u8 + 8;
        }
        Operation::WaveBallot { rd, ps } => {
            inst.opcode = Opcode::WaveOp;
            inst.rd = *rd;
            inst.rs1 = *ps;
            inst.modifier = WaveOpType::Ballot as u8;
        }
        Operation::WaveVote { op, pd, ps } => {
            inst.opcode = Opcode::WaveOp;
            inst.rd = *pd;
            inst.rs1 = *ps;
            inst.modifier = *op as u8;
        }
        Operation::MmaLoadA { rd, rs1, rs2 } => {
            inst.opcode = Opcode::Mma;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
            inst.modifier = MmaOp::LoadA as u8;
        }
        Operation::MmaLoadB { rd, rs1, rs2 } => {
            inst.opcode = Opcode::Mma;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
            inst.modifier = MmaOp::LoadB as u8;
        }
        Operation::MmaStoreC { rd, rs1, rs2 } => {
            inst.opcode = Opcode::Mma;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
            inst.modifier = MmaOp::StoreC as u8;
        }
        Operation::MmaCompute { rd, rs1, rs2 } => {
            inst.opcode = Opcode::Mma;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.rs2 = *rs2;
            inst.modifier = MmaOp::Compute as u8;
        }
        _ => unreachable!("convert_instruction routes only wave operations here"),
    }
}

fn convert_control(op: &wave_decode::Operation, inst: &mut DecodedInstruction) {
    use wave_decode::Operation;
    match op {
        Operation::If { ps } => {
            inst.opcode = Opcode::Control;
            inst.rs1 = *ps;
            inst.modifier = ControlOp::If as u8;
        }
        Operation::Else => {
            inst.opcode = Opcode::Control;
            inst.modifier = ControlOp::Else as u8;
        }
        Operation::Endif => {
            inst.opcode = Opcode::Control;
            inst.modifier = ControlOp::Endif as u8;
        }
        Operation::Loop => {
            inst.opcode = Opcode::Control;
            inst.modifier = ControlOp::Loop as u8;
        }
        Operation::Break { ps } => {
            inst.opcode = Opcode::Control;
            inst.rs1 = *ps;
            inst.modifier = ControlOp::Break as u8;
        }
        Operation::Continue { ps } => {
            inst.opcode = Opcode::Control;
            inst.rs1 = *ps;
            inst.modifier = ControlOp::Continue as u8;
        }
        Operation::Endloop => {
            inst.opcode = Opcode::Control;
            inst.modifier = ControlOp::Endloop as u8;
        }
        Operation::Call { target } => {
            inst.opcode = Opcode::Control;
            inst.modifier = ControlOp::Call as u8;
            inst.immediate = *target;
        }
        Operation::Return => {
            inst.opcode = Opcode::Control;
            inst.modifier = SyncOp::Return as u8 + SYNC_MODIFIER_OFFSET;
        }
        Operation::Halt => {
            inst.opcode = Opcode::Control;
            inst.modifier = SyncOp::Halt as u8 + SYNC_MODIFIER_OFFSET;
        }
        Operation::Barrier => {
            inst.opcode = Opcode::Control;
            inst.modifier = SyncOp::Barrier as u8 + SYNC_MODIFIER_OFFSET;
        }
        Operation::FenceAcquire { scope } => {
            inst.opcode = Opcode::Control;
            inst.modifier = SyncOp::FenceAcquire as u8 + SYNC_MODIFIER_OFFSET;
            inst.scope = *scope as u8;
        }
        Operation::FenceRelease { scope } => {
            inst.opcode = Opcode::Control;
            inst.modifier = SyncOp::FenceRelease as u8 + SYNC_MODIFIER_OFFSET;
            inst.scope = *scope as u8;
        }
        Operation::FenceAcqRel { scope } => {
            inst.opcode = Opcode::Control;
            inst.modifier = SyncOp::FenceAcqRel as u8 + SYNC_MODIFIER_OFFSET;
            inst.scope = *scope as u8;
        }
        Operation::Wait => {
            inst.opcode = Opcode::Control;
            inst.modifier = SyncOp::Wait as u8 + SYNC_MODIFIER_OFFSET;
        }
        Operation::Nop => {
            inst.opcode = Opcode::Control;
            inst.modifier = SyncOp::Nop as u8 + SYNC_MODIFIER_OFFSET;
        }
        _ => unreachable!("convert_instruction routes only control operations here"),
    }
}

fn convert_move(op: &wave_decode::Operation, inst: &mut DecodedInstruction) {
    use wave_decode::Operation;
    match op {
        Operation::Mov { rd, rs1 } => {
            inst.opcode = Opcode::Misc;
            inst.rd = *rd;
            inst.rs1 = *rs1;
            inst.modifier = MiscOp::Mov as u8;
        }
        Operation::MovImm { rd, imm } => {
            inst.opcode = Opcode::Misc;
            inst.rd = *rd;
            inst.modifier = MiscOp::MovImm as u8;
            inst.immediate = *imm;
        }
        Operation::MovSr { rd, sr_index } => {
            inst.opcode = Opcode::Misc;
            inst.rd = *rd;
            inst.rs1 = *sr_index;
            inst.modifier = MiscOp::MovSr as u8;
        }
        Operation::Unknown {
            opcode,
            word0,
            word1: _,
        } => {
            let opc = Opcode::from_u8(*opcode).unwrap_or(Opcode::Control);
            inst.opcode = opc;
            inst.immediate = *word0;
        }
        _ => unreachable!("convert_instruction routes only move operations here"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_word0(opcode: u8, rd: u8, rs1: u8, modifier: u8, pred: u8) -> Vec<u8> {
        let word = ((u32::from(opcode) & 0xFF) << 24)
            | ((u32::from(rd) & 0xFF) << 16)
            | ((u32::from(rs1) & 0xFF) << 8)
            | ((u32::from(modifier) & 0x0F) << 4)
            | u32::from(pred & 0x07);
        word.to_le_bytes().to_vec()
    }

    fn encode_ext_rs2(rs2: u8) -> Vec<u8> {
        let word1 = (u32::from(rs2) & 0xFF) << 24;
        word1.to_le_bytes().to_vec()
    }

    #[test]
    fn test_decoder_iadd() {
        let mut code = encode_word0(0x00, 1, 2, 0, 0);
        code.extend_from_slice(&encode_ext_rs2(3));
        let decoder = Decoder::new(&code);
        let inst = decoder.decode_at(0).unwrap();

        assert_eq!(inst.opcode, Opcode::Iadd);
        assert_eq!(inst.rd, 1);
        assert_eq!(inst.rs1, 2);
        assert_eq!(inst.rs2, 3);
        assert_eq!(inst.size, 8);
    }

    #[test]
    fn test_decoder_halt() {
        let code = encode_word0(0x3F, 0, 0, SyncOp::Halt as u8 + SYNC_MODIFIER_OFFSET, 0);
        let decoder = Decoder::new(&code);
        let inst = decoder.decode_at(0).unwrap();

        assert_eq!(inst.opcode, Opcode::Control);
        assert!(inst.is_sync_op());
        assert_eq!(inst.modifier, SyncOp::Halt as u8 + SYNC_MODIFIER_OFFSET);
    }

    #[test]
    fn test_decoder_mov_imm() {
        let mut code = encode_word0(0x41, 5, 0, MiscOp::MovImm as u8, 0);
        let word1 = 0x1234_5678u32;
        code.extend_from_slice(&word1.to_le_bytes());

        let decoder = Decoder::new(&code);
        let inst = decoder.decode_at(0).unwrap();

        assert_eq!(inst.opcode, Opcode::Misc);
        assert!(inst.is_misc_op());
        assert_eq!(inst.rd, 5);
        assert_eq!(inst.immediate, 0x1234_5678);
        assert_eq!(inst.size, 8);
    }

    #[test]
    fn test_decoder_disassemble() {
        let mut code = encode_word0(0x00, 1, 2, 0, 0);
        code.extend_from_slice(&encode_ext_rs2(3));
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
