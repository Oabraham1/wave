// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Core instruction decoder. Reads binary instruction words and produces
//!
//! structured `DecodedInstruction` values.

use crate::instruction::{DecodedInstruction, Operation};
use crate::opcodes::{
    AtomicOp, Bf16Op, Bf16PackedOp, BitOpType, CmpOp, ControlOp, CvtType, F16Op, F16PackedOp,
    F64DivSqrtOp, F64Op, FUnaryOp, MemWidth, MiscOp, MmaOp, Opcode, Scope, SyncOp, WaveOpType,
    WaveReduceType, EXTENDED_RS2_MASK, EXTENDED_RS2_SHIFT, EXTENDED_RS3_MASK, EXTENDED_RS3_SHIFT,
    EXTENDED_RS4_MASK, EXTENDED_RS4_SHIFT, EXTENDED_SCOPE_MASK, EXTENDED_SCOPE_SHIFT,
    MODIFIER_MASK, MODIFIER_SHIFT, OPCODE_MASK, OPCODE_SHIFT, PRED_NEG_MASK, PRED_NEG_SHIFT,
    PRED_REG_MASK, PRED_REG_SHIFT, RD_MASK, RD_SHIFT, RS1_MASK, RS1_SHIFT, SYNC_MODIFIER_OFFSET,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DecodeError {
    #[error("unexpected end of code at offset {offset}")]
    UnexpectedEnd { offset: u32 },
    #[error("invalid opcode 0x{opcode:02x} at offset {offset}")]
    InvalidOpcode { opcode: u8, offset: u32 },
    #[error("invalid modifier {modifier} for opcode {opcode:?} at offset {offset}")]
    InvalidModifier {
        opcode: Opcode,
        modifier: u8,
        offset: u32,
    },
}

/// WAVE instruction decoder
pub struct Decoder<'a> {
    code: &'a [u8],
    offset: u32,
}

impl<'a> Decoder<'a> {
    /// Create a new decoder for the given code bytes
    #[must_use]
    pub fn new(code: &'a [u8]) -> Self {
        Self { code, offset: 0 }
    }

    /// Get current offset in bytes
    #[must_use]
    pub fn offset(&self) -> u32 {
        self.offset
    }

    /// Check if there are more instructions to decode
    #[must_use]
    pub fn has_more(&self) -> bool {
        (self.offset as usize) + 4 <= self.code.len()
    }

    /// Decode the next instruction
    ///
    /// # Errors
    ///
    /// Returns `DecodeError::UnexpectedEnd` if there are fewer than 4 bytes remaining.
    /// Returns `DecodeError::InvalidOpcode` if the opcode byte is not recognized.
    /// Returns `DecodeError::InvalidModifier` if the modifier is invalid for the opcode.
    #[allow(clippy::similar_names)]
    pub fn decode_next(&mut self) -> Result<DecodedInstruction, DecodeError> {
        let offset = self.offset;

        if (offset as usize) + 4 > self.code.len() {
            return Err(DecodeError::UnexpectedEnd { offset });
        }

        let word0 = self.read_u32();
        let opcode_raw = ((word0 >> OPCODE_SHIFT) & OPCODE_MASK) as u8;
        let rd = ((word0 >> RD_SHIFT) & RD_MASK) as u8;
        let rs1 = ((word0 >> RS1_SHIFT) & RS1_MASK) as u8;
        let modifier = ((word0 >> MODIFIER_SHIFT) & MODIFIER_MASK) as u8;
        let pred_reg = ((word0 >> PRED_REG_SHIFT) & PRED_REG_MASK) as u8;
        let pred_neg = ((word0 >> PRED_NEG_SHIFT) & PRED_NEG_MASK) != 0;

        let opcode = Opcode::from_u8(opcode_raw).ok_or(DecodeError::InvalidOpcode {
            opcode: opcode_raw,
            offset,
        })?;

        let operation = self.decode_operation(opcode, rd, rs1, modifier, offset)?;

        let size = if self.offset > offset + 4 { 8 } else { 4 };

        Ok(DecodedInstruction {
            offset,
            size,
            operation,
            predicate: pred_reg,
            predicate_negated: pred_neg,
        })
    }

    fn read_u32(&mut self) -> u32 {
        let off = self.offset as usize;
        let bytes = &self.code[off..off + 4];
        self.offset += 4;
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn decode_operation(
        &mut self,
        opcode: Opcode,
        rd: u8,
        rs1: u8,
        modifier: u8,
        offset: u32,
    ) -> Result<Operation, DecodeError> {
        let op = match opcode {
            Opcode::Iadd => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::Iadd { rd, rs1, rs2 }
            }
            Opcode::Isub => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::Isub { rd, rs1, rs2 }
            }
            Opcode::Imul => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::Imul { rd, rs1, rs2 }
            }
            Opcode::ImulHi => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::ImulHi { rd, rs1, rs2 }
            }
            Opcode::Imad => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                let rs3 = ((word1 >> EXTENDED_RS3_SHIFT) & EXTENDED_RS3_MASK) as u8;
                Operation::Imad { rd, rs1, rs2, rs3 }
            }
            Opcode::Idiv => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::Idiv { rd, rs1, rs2 }
            }
            Opcode::Imod => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::Imod { rd, rs1, rs2 }
            }
            Opcode::Ineg => Operation::Ineg { rd, rs1 },
            Opcode::Iabs => Operation::Iabs { rd, rs1 },
            Opcode::Imin => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::Imin { rd, rs1, rs2 }
            }
            Opcode::Imax => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::Imax { rd, rs1, rs2 }
            }
            Opcode::Iclamp => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                let rs3 = ((word1 >> EXTENDED_RS3_SHIFT) & EXTENDED_RS3_MASK) as u8;
                Operation::Iclamp { rd, rs1, rs2, rs3 }
            }

            Opcode::Fadd => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::Fadd { rd, rs1, rs2 }
            }
            Opcode::Fsub => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::Fsub { rd, rs1, rs2 }
            }
            Opcode::Fmul => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::Fmul { rd, rs1, rs2 }
            }
            Opcode::Fma => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                let rs3 = ((word1 >> EXTENDED_RS3_SHIFT) & EXTENDED_RS3_MASK) as u8;
                Operation::Fma { rd, rs1, rs2, rs3 }
            }
            Opcode::Fdiv => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::Fdiv { rd, rs1, rs2 }
            }
            Opcode::Fneg => Operation::Fneg { rd, rs1 },
            Opcode::Fabs => Operation::Fabs { rd, rs1 },
            Opcode::Fmin => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::Fmin { rd, rs1, rs2 }
            }
            Opcode::Fmax => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::Fmax { rd, rs1, rs2 }
            }
            Opcode::Fclamp => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                let rs3 = ((word1 >> EXTENDED_RS3_SHIFT) & EXTENDED_RS3_MASK) as u8;
                Operation::Fclamp { rd, rs1, rs2, rs3 }
            }
            Opcode::Fsqrt => Operation::Fsqrt { rd, rs1 },
            Opcode::FUnaryOps => {
                let op = FUnaryOp::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                Operation::FUnary { op, rd, rs1 }
            }

            Opcode::F16Ops => {
                let op = F16Op::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                let rs3 = if op == F16Op::Hma {
                    Some(((word1 >> EXTENDED_RS3_SHIFT) & EXTENDED_RS3_MASK) as u8)
                } else {
                    None
                };
                Operation::F16 {
                    op,
                    rd,
                    rs1,
                    rs2,
                    rs3,
                }
            }

            Opcode::F16PackedOps => {
                let op = F16PackedOp::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                let rs3 = if op == F16PackedOp::Hma2 {
                    Some(((word1 >> EXTENDED_RS3_SHIFT) & EXTENDED_RS3_MASK) as u8)
                } else {
                    None
                };
                Operation::F16Packed {
                    op,
                    rd,
                    rs1,
                    rs2,
                    rs3,
                }
            }

            Opcode::Bf16Ops => {
                let op = Bf16Op::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                let rs3 = if op == Bf16Op::Bma {
                    Some(((word1 >> EXTENDED_RS3_SHIFT) & EXTENDED_RS3_MASK) as u8)
                } else {
                    None
                };
                Operation::Bf16 {
                    op,
                    rd,
                    rs1,
                    rs2,
                    rs3,
                }
            }

            Opcode::Bf16PackedOps => {
                let op = Bf16PackedOp::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                let rs3 = if op == Bf16PackedOp::Bma2 {
                    Some(((word1 >> EXTENDED_RS3_SHIFT) & EXTENDED_RS3_MASK) as u8)
                } else {
                    None
                };
                Operation::Bf16Packed {
                    op,
                    rd,
                    rs1,
                    rs2,
                    rs3,
                }
            }

            Opcode::F64Ops => {
                let op = F64Op::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                let rs3 = if op == F64Op::Dma {
                    Some(((word1 >> EXTENDED_RS3_SHIFT) & EXTENDED_RS3_MASK) as u8)
                } else {
                    None
                };
                Operation::F64 {
                    op,
                    rd,
                    rs1,
                    rs2,
                    rs3,
                }
            }

            Opcode::F64DivSqrt => {
                let op = F64DivSqrtOp::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                let rs2_opt = if op == F64DivSqrtOp::Ddiv {
                    let word1 = self.read_u32();
                    Some(((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8)
                } else {
                    None
                };
                Operation::F64DivSqrt {
                    op,
                    rd,
                    rs1,
                    rs2: rs2_opt,
                }
            }

            Opcode::And => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::And { rd, rs1, rs2 }
            }
            Opcode::Or => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::Or { rd, rs1, rs2 }
            }
            Opcode::Xor => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::Xor { rd, rs1, rs2 }
            }
            Opcode::Not => Operation::Not { rd, rs1 },
            Opcode::Shl => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::Shl { rd, rs1, rs2 }
            }
            Opcode::Shr => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::Shr { rd, rs1, rs2 }
            }
            Opcode::Sar => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::Sar { rd, rs1, rs2 }
            }
            Opcode::BitOps => {
                let op = BitOpType::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                match op {
                    BitOpType::Bitcount | BitOpType::Bitfind | BitOpType::Bitrev => {
                        Operation::BitOp {
                            op,
                            rd,
                            rs1,
                            rs2: None,
                            rs3: None,
                            rs4: None,
                        }
                    }
                    BitOpType::Bfe => {
                        let word1 = self.read_u32();
                        let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                        let rs3 = ((word1 >> EXTENDED_RS3_SHIFT) & EXTENDED_RS3_MASK) as u8;
                        Operation::BitOp {
                            op,
                            rd,
                            rs1,
                            rs2: Some(rs2),
                            rs3: Some(rs3),
                            rs4: None,
                        }
                    }
                    BitOpType::Bfi => {
                        let word1 = self.read_u32();
                        let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                        let rs3 = ((word1 >> EXTENDED_RS3_SHIFT) & EXTENDED_RS3_MASK) as u8;
                        let rs4 = ((word1 >> EXTENDED_RS4_SHIFT) & EXTENDED_RS4_MASK) as u8;
                        Operation::BitOp {
                            op,
                            rd,
                            rs1,
                            rs2: Some(rs2),
                            rs3: Some(rs3),
                            rs4: Some(rs4),
                        }
                    }
                }
            }

            Opcode::Icmp => {
                let op = CmpOp::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::Icmp {
                    op,
                    pd: rd,
                    rs1,
                    rs2,
                }
            }
            Opcode::Ucmp => {
                let op = CmpOp::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::Ucmp {
                    op,
                    pd: rd,
                    rs1,
                    rs2,
                }
            }
            Opcode::Fcmp => {
                let op = CmpOp::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::Fcmp {
                    op,
                    pd: rd,
                    rs1,
                    rs2,
                }
            }

            Opcode::Select => {
                let word1 = self.read_u32();
                let sel_rs1 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                let sel_rs2 = ((word1 >> EXTENDED_RS3_SHIFT) & EXTENDED_RS3_MASK) as u8;
                Operation::Select {
                    rd,
                    ps: rs1,
                    rs1: sel_rs1,
                    rs2: sel_rs2,
                }
            }
            Opcode::Cvt => {
                let cvt_type = CvtType::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                Operation::Cvt { cvt_type, rd, rs1 }
            }

            Opcode::LocalLoad => {
                let width = MemWidth::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                Operation::LocalLoad {
                    width,
                    rd,
                    addr: rs1,
                }
            }
            Opcode::LocalStore => {
                let width = MemWidth::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::LocalStore {
                    width,
                    addr: rs1,
                    value: rs2,
                }
            }

            Opcode::DeviceLoad => {
                let width = MemWidth::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                Operation::DeviceLoad {
                    width,
                    rd,
                    addr: rs1,
                }
            }
            Opcode::DeviceStore => {
                let width = MemWidth::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                Operation::DeviceStore {
                    width,
                    addr: rs1,
                    value: rs2,
                }
            }

            Opcode::LocalAtomic => {
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                if modifier == 8 {
                    let rs3 = ((word1 >> EXTENDED_RS3_SHIFT) & EXTENDED_RS3_MASK) as u8;
                    let rd_opt = if rd != 0 { Some(rd) } else { None };
                    Operation::LocalAtomicCas {
                        rd: rd_opt,
                        addr: rs1,
                        expected: rs2,
                        desired: rs3,
                    }
                } else {
                    let op = AtomicOp::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                        opcode,
                        modifier,
                        offset,
                    })?;
                    let rd_opt = if rd != 0 { Some(rd) } else { None };
                    Operation::LocalAtomic {
                        op,
                        rd: rd_opt,
                        addr: rs1,
                        value: rs2,
                    }
                }
            }

            Opcode::DeviceAtomic => {
                let word1 = self.read_u32();
                let ext_scope = ((word1 >> EXTENDED_SCOPE_SHIFT) & EXTENDED_SCOPE_MASK) as u8;
                let scope_val = Scope::from_u8(ext_scope).unwrap_or(Scope::Device);
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                if modifier == 8 {
                    let rs3 = ((word1 >> EXTENDED_RS3_SHIFT) & EXTENDED_RS3_MASK) as u8;
                    let rd_opt = if rd != 0 { Some(rd) } else { None };
                    Operation::DeviceAtomicCas {
                        rd: rd_opt,
                        addr: rs1,
                        expected: rs2,
                        desired: rs3,
                        scope: scope_val,
                    }
                } else {
                    let op = AtomicOp::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                        opcode,
                        modifier,
                        offset,
                    })?;
                    let rd_opt = if rd != 0 { Some(rd) } else { None };
                    Operation::DeviceAtomic {
                        op,
                        rd: rd_opt,
                        addr: rs1,
                        value: rs2,
                        scope: scope_val,
                    }
                }
            }

            Opcode::WaveOp => {
                if modifier >= 8 {
                    let op = WaveReduceType::from_u8(modifier - 8).ok_or(
                        DecodeError::InvalidModifier {
                            opcode,
                            modifier,
                            offset,
                        },
                    )?;
                    Operation::WaveReduce { op, rd, rs1 }
                } else {
                    let op = WaveOpType::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                        opcode,
                        modifier,
                        offset,
                    })?;
                    match op {
                        WaveOpType::Ballot => Operation::WaveBallot { rd, ps: rs1 },
                        WaveOpType::Any | WaveOpType::All => Operation::WaveVote {
                            op,
                            pd: rd,
                            ps: rs1,
                        },
                        _ => {
                            let word1 = self.read_u32();
                            let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                            Operation::WaveOp {
                                op,
                                rd,
                                rs1,
                                rs2: Some(rs2),
                            }
                        }
                    }
                }
            }

            Opcode::Control => self.decode_control(rd, rs1, modifier, offset)?,

            Opcode::Misc => self.decode_misc(rd, rs1, modifier, offset)?,

            Opcode::Mma => {
                let mma_op = MmaOp::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                let word1 = self.read_u32();
                let rs2 = ((word1 >> EXTENDED_RS2_SHIFT) & EXTENDED_RS2_MASK) as u8;
                match mma_op {
                    MmaOp::LoadA => Operation::MmaLoadA { rd, rs1, rs2 },
                    MmaOp::LoadB => Operation::MmaLoadB { rd, rs1, rs2 },
                    MmaOp::StoreC => Operation::MmaStoreC { rd, rs1, rs2 },
                    MmaOp::Compute => Operation::MmaCompute { rd, rs1, rs2 },
                }
            }
        };

        Ok(op)
    }

    fn decode_control(
        &mut self,
        _rd: u8,
        rs1: u8,
        modifier: u8,
        offset: u32,
    ) -> Result<Operation, DecodeError> {
        if modifier >= SYNC_MODIFIER_OFFSET {
            let sync_mod = modifier - SYNC_MODIFIER_OFFSET;
            let op = SyncOp::from_u8(sync_mod).ok_or(DecodeError::InvalidModifier {
                opcode: Opcode::Control,
                modifier,
                offset,
            })?;
            match op {
                SyncOp::Return => Ok(Operation::Return),
                SyncOp::Halt => Ok(Operation::Halt),
                SyncOp::Barrier => Ok(Operation::Barrier),
                SyncOp::FenceAcquire => {
                    let word1 = self.read_u32();
                    let ext_scope = ((word1 >> EXTENDED_SCOPE_SHIFT) & EXTENDED_SCOPE_MASK) as u8;
                    let scope_val = Scope::from_u8(ext_scope).unwrap_or(Scope::Workgroup);
                    Ok(Operation::FenceAcquire { scope: scope_val })
                }
                SyncOp::FenceRelease => {
                    let word1 = self.read_u32();
                    let ext_scope = ((word1 >> EXTENDED_SCOPE_SHIFT) & EXTENDED_SCOPE_MASK) as u8;
                    let scope_val = Scope::from_u8(ext_scope).unwrap_or(Scope::Workgroup);
                    Ok(Operation::FenceRelease { scope: scope_val })
                }
                SyncOp::FenceAcqRel => {
                    let word1 = self.read_u32();
                    let ext_scope = ((word1 >> EXTENDED_SCOPE_SHIFT) & EXTENDED_SCOPE_MASK) as u8;
                    let scope_val = Scope::from_u8(ext_scope).unwrap_or(Scope::Workgroup);
                    Ok(Operation::FenceAcqRel { scope: scope_val })
                }
                SyncOp::Wait => Ok(Operation::Wait),
                SyncOp::Nop => Ok(Operation::Nop),
            }
        } else {
            let op = ControlOp::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                opcode: Opcode::Control,
                modifier,
                offset,
            })?;
            match op {
                ControlOp::If => Ok(Operation::If { ps: rs1 }),
                ControlOp::Else => Ok(Operation::Else),
                ControlOp::Endif => Ok(Operation::Endif),
                ControlOp::Loop => Ok(Operation::Loop),
                ControlOp::Break => Ok(Operation::Break { ps: rs1 }),
                ControlOp::Continue => Ok(Operation::Continue { ps: rs1 }),
                ControlOp::Endloop => Ok(Operation::Endloop),
                ControlOp::Call => {
                    let target = self.read_u32();
                    Ok(Operation::Call { target })
                }
            }
        }
    }

    fn decode_misc(
        &mut self,
        rd: u8,
        rs1: u8,
        modifier: u8,
        offset: u32,
    ) -> Result<Operation, DecodeError> {
        let op = MiscOp::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
            opcode: Opcode::Misc,
            modifier,
            offset,
        })?;
        match op {
            MiscOp::Mov => Ok(Operation::Mov { rd, rs1 }),
            MiscOp::MovImm => {
                let imm = self.read_u32();
                Ok(Operation::MovImm { rd, imm })
            }
            MiscOp::MovSr => Ok(Operation::MovSr { rd, sr_index: rs1 }),
        }
    }
}

/// Decode a single instruction at the given offset
///
/// # Errors
///
/// Returns a `DecodeError` if the instruction cannot be decoded.
pub fn decode_at(code: &[u8], offset: u32) -> Result<DecodedInstruction, DecodeError> {
    let mut decoder = Decoder::new(code);
    decoder.offset = offset;
    decoder.decode_next()
}

/// Decode all instructions in the code section
///
/// # Errors
///
/// Returns a `DecodeError` if any instruction cannot be decoded.
pub fn decode_all(code: &[u8]) -> Result<Vec<DecodedInstruction>, DecodeError> {
    let mut decoder = Decoder::new(code);
    let mut instructions = Vec::new();

    while decoder.has_more() {
        instructions.push(decoder.decode_next()?);
    }

    Ok(instructions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_base(opcode: u8, rd: u8, rs1: u8, modifier: u8, pred: u8) -> [u8; 4] {
        let word = ((u32::from(opcode) & OPCODE_MASK) << OPCODE_SHIFT)
            | ((u32::from(rd) & RD_MASK) << RD_SHIFT)
            | ((u32::from(rs1) & RS1_MASK) << RS1_SHIFT)
            | ((u32::from(modifier) & MODIFIER_MASK) << MODIFIER_SHIFT)
            | u32::from(pred & 0x07);
        word.to_le_bytes()
    }

    fn encode_extended_rs2(rs2: u8) -> [u8; 4] {
        let word1 = (u32::from(rs2) & EXTENDED_RS2_MASK) << EXTENDED_RS2_SHIFT;
        word1.to_le_bytes()
    }

    #[test]
    fn test_decode_iadd() {
        let mut code = encode_base(0x00, 5, 3, 0, 0).to_vec();
        code.extend_from_slice(&encode_extended_rs2(4));
        let instr = decode_at(&code, 0).unwrap();
        assert_eq!(instr.size, 8);
        assert_eq!(
            instr.operation,
            Operation::Iadd {
                rd: 5,
                rs1: 3,
                rs2: 4
            }
        );
    }

    #[test]
    fn test_decode_halt() {
        let code = encode_base(0x3F, 0, 0, SyncOp::Halt as u8 + SYNC_MODIFIER_OFFSET, 0);
        let instr = decode_at(&code, 0).unwrap();
        assert_eq!(instr.operation, Operation::Halt);
    }

    #[test]
    fn test_decode_mov_imm() {
        let mut code = encode_base(0x41, 5, 0, MiscOp::MovImm as u8, 0).to_vec();
        code.extend_from_slice(&0x12345678u32.to_le_bytes());

        let instr = decode_at(&code, 0).unwrap();
        assert_eq!(instr.size, 8);
        assert_eq!(
            instr.operation,
            Operation::MovImm {
                rd: 5,
                imm: 0x12345678
            }
        );
    }

    #[test]
    fn test_decode_mov_sr() {
        let code = encode_base(0x41, 5, 4, MiscOp::MovSr as u8, 0);
        let instr = decode_at(&code, 0).unwrap();
        assert_eq!(instr.operation, Operation::MovSr { rd: 5, sr_index: 4 });
    }

    #[test]
    fn test_decode_device_store_u32() {
        let mut code = encode_base(0x39, 0, 3, MemWidth::U32 as u8, 0).to_vec();
        code.extend_from_slice(&encode_extended_rs2(5));
        let instr = decode_at(&code, 0).unwrap();
        assert_eq!(
            instr.operation,
            Operation::DeviceStore {
                width: MemWidth::U32,
                addr: 3,
                value: 5
            }
        );
    }

    #[test]
    fn test_decode_all() {
        let mut code = encode_base(0x00, 5, 3, 0, 0).to_vec(); // iadd base word
        code.extend_from_slice(&encode_extended_rs2(4)); // iadd extended word
        code.extend_from_slice(&encode_base(
            0x3F,
            0,
            0,
            SyncOp::Halt as u8 + SYNC_MODIFIER_OFFSET,
            0,
        )); // halt

        let instructions = decode_all(&code).unwrap();
        assert_eq!(instructions.len(), 2);
        assert_eq!(instructions[0].offset, 0);
        assert_eq!(instructions[1].offset, 8);
    }

    #[test]
    fn test_decode_shl() {
        let mut code = encode_base(0x24, 2, 0, 0, 0).to_vec();
        code.extend_from_slice(&encode_extended_rs2(1));
        let instr = decode_at(&code, 0).unwrap();
        assert_eq!(
            instr.operation,
            Operation::Shl {
                rd: 2,
                rs1: 0,
                rs2: 1
            }
        );
    }

    #[test]
    fn test_decode_predication() {
        let pred_bits: u8 = 1 | (1 << 2);
        let mut code = encode_base(0x00, 5, 3, 0, pred_bits).to_vec();
        code.extend_from_slice(&encode_extended_rs2(4));
        let instr = decode_at(&code, 0).unwrap();
        assert_eq!(instr.predicate, 1);
        assert!(instr.predicate_negated);
    }
}
