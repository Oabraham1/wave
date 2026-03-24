// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Core instruction decoder. Reads binary instruction words and produces
//!
//! structured DecodedInstruction values.

use crate::instruction::{DecodedInstruction, Operation};
use crate::opcodes::{
    AtomicOp, BitOpType, CmpOp, ControlOp, CvtType, F16Op, F16PackedOp, F64DivSqrtOp, F64Op,
    FUnaryOp, MemWidth, MiscOp, Opcode, Scope, SyncOp, WaveOpType, WaveReduceType,
    EXTENDED_RS3_MASK, EXTENDED_RS3_SHIFT, EXTENDED_RS4_MASK, EXTENDED_RS4_SHIFT, FLAGS_MASK,
    MISC_OP_FLAG, MODIFIER_MASK, MODIFIER_SHIFT, OPCODE_MASK, OPCODE_SHIFT, PRED_MASK,
    PRED_NEG_MASK, PRED_NEG_SHIFT, PRED_SHIFT, RD_MASK, RD_SHIFT, RS1_MASK, RS1_SHIFT, RS2_MASK,
    RS2_SHIFT, SCOPE_MASK, SCOPE_SHIFT, SYNC_OP_FLAG,
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
    pub fn decode_next(&mut self) -> Result<DecodedInstruction, DecodeError> {
        let offset = self.offset;

        if (offset as usize) + 4 > self.code.len() {
            return Err(DecodeError::UnexpectedEnd { offset });
        }

        let word0 = self.read_u32();
        let opcode_raw = ((word0 >> OPCODE_SHIFT) & OPCODE_MASK) as u8;
        let rd = ((word0 >> RD_SHIFT) & RD_MASK) as u8;
        let rs1 = ((word0 >> RS1_SHIFT) & RS1_MASK) as u8;
        let rs2 = ((word0 >> RS2_SHIFT) & RS2_MASK) as u8;
        let modifier = ((word0 >> MODIFIER_SHIFT) & MODIFIER_MASK) as u8;
        let scope = ((word0 >> SCOPE_SHIFT) & SCOPE_MASK) as u8;
        let predicate = ((word0 >> PRED_SHIFT) & PRED_MASK) as u8;
        let pred_neg = ((word0 >> PRED_NEG_SHIFT) & PRED_NEG_MASK) != 0;
        let flags = (word0 & FLAGS_MASK) as u8;

        let opcode = Opcode::from_u8(opcode_raw).ok_or(DecodeError::InvalidOpcode {
            opcode: opcode_raw,
            offset,
        })?;

        let operation = self.decode_operation(
            opcode, rd, rs1, rs2, modifier, scope, flags, offset,
        )?;

        let size = if self.offset > offset + 4 { 8 } else { 4 };

        Ok(DecodedInstruction {
            offset,
            size,
            operation,
            predicate,
            predicate_negated: pred_neg,
        })
    }

    fn read_u32(&mut self) -> u32 {
        let off = self.offset as usize;
        let bytes = &self.code[off..off + 4];
        self.offset += 4;
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    fn peek_u32(&self) -> Option<u32> {
        let off = self.offset as usize;
        if off + 4 <= self.code.len() {
            let bytes = &self.code[off..off + 4];
            Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        } else {
            None
        }
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn decode_operation(
        &mut self,
        opcode: Opcode,
        rd: u8,
        rs1: u8,
        rs2: u8,
        modifier: u8,
        scope: u8,
        flags: u8,
        offset: u32,
    ) -> Result<Operation, DecodeError> {
        let op = match opcode {

            Opcode::Iadd => Operation::Iadd { rd, rs1, rs2 },
            Opcode::Isub => Operation::Isub { rd, rs1, rs2 },
            Opcode::Imul => Operation::Imul { rd, rs1, rs2 },
            Opcode::ImulHi => Operation::ImulHi { rd, rs1, rs2 },
            Opcode::Imad => {
                let word1 = self.read_u32();
                let rs3 = ((word1 >> EXTENDED_RS3_SHIFT) & EXTENDED_RS3_MASK) as u8;
                Operation::Imad { rd, rs1, rs2, rs3 }
            }
            Opcode::Idiv => Operation::Idiv { rd, rs1, rs2 },
            Opcode::Imod => Operation::Imod { rd, rs1, rs2 },
            Opcode::Ineg => Operation::Ineg { rd, rs1 },
            Opcode::Iabs => Operation::Iabs { rd, rs1 },
            Opcode::Imin => Operation::Imin { rd, rs1, rs2 },
            Opcode::Imax => Operation::Imax { rd, rs1, rs2 },
            Opcode::Iclamp => {
                let word1 = self.read_u32();
                let rs3 = ((word1 >> EXTENDED_RS3_SHIFT) & EXTENDED_RS3_MASK) as u8;
                Operation::Iclamp { rd, rs1, rs2, rs3 }
            }

            Opcode::Fadd => Operation::Fadd { rd, rs1, rs2 },
            Opcode::Fsub => Operation::Fsub { rd, rs1, rs2 },
            Opcode::Fmul => Operation::Fmul { rd, rs1, rs2 },
            Opcode::Fma => {
                let word1 = self.read_u32();
                let rs3 = ((word1 >> EXTENDED_RS3_SHIFT) & EXTENDED_RS3_MASK) as u8;
                Operation::Fma { rd, rs1, rs2, rs3 }
            }
            Opcode::Fdiv => Operation::Fdiv { rd, rs1, rs2 },
            Opcode::Fneg => Operation::Fneg { rd, rs1 },
            Opcode::Fabs => Operation::Fabs { rd, rs1 },
            Opcode::Fmin => Operation::Fmin { rd, rs1, rs2 },
            Opcode::Fmax => Operation::Fmax { rd, rs1, rs2 },
            Opcode::Fclamp => {
                let word1 = self.read_u32();
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
                let rs3 = if op == F16Op::Hma {
                    let word1 = self.read_u32();
                    Some(((word1 >> EXTENDED_RS3_SHIFT) & EXTENDED_RS3_MASK) as u8)
                } else {
                    None
                };
                Operation::F16 { op, rd, rs1, rs2, rs3 }
            }

            Opcode::F16PackedOps => {
                let op = F16PackedOp::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                let rs3 = if op == F16PackedOp::Hma2 {
                    let word1 = self.read_u32();
                    Some(((word1 >> EXTENDED_RS3_SHIFT) & EXTENDED_RS3_MASK) as u8)
                } else {
                    None
                };
                Operation::F16Packed { op, rd, rs1, rs2, rs3 }
            }

            Opcode::F64Ops => {
                let op = F64Op::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                let rs3 = if op == F64Op::Dma {
                    let word1 = self.read_u32();
                    Some(((word1 >> EXTENDED_RS3_SHIFT) & EXTENDED_RS3_MASK) as u8)
                } else {
                    None
                };
                Operation::F64 { op, rd, rs1, rs2, rs3 }
            }

            Opcode::F64DivSqrt => {
                let op = F64DivSqrtOp::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                let rs2_opt = if op == F64DivSqrtOp::Ddiv {
                    Some(rs2)
                } else {
                    None
                };
                Operation::F64DivSqrt { op, rd, rs1, rs2: rs2_opt }
            }

            Opcode::And => Operation::And { rd, rs1, rs2 },
            Opcode::Or => Operation::Or { rd, rs1, rs2 },
            Opcode::Xor => Operation::Xor { rd, rs1, rs2 },
            Opcode::Not => Operation::Not { rd, rs1 },
            Opcode::Shl => Operation::Shl { rd, rs1, rs2 },
            Opcode::Shr => Operation::Shr { rd, rs1, rs2 },
            Opcode::Sar => Operation::Sar { rd, rs1, rs2 },
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
                Operation::Icmp { op, pd: rd, rs1, rs2 }
            }
            Opcode::Ucmp => {
                let op = CmpOp::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                Operation::Ucmp { op, pd: rd, rs1, rs2 }
            }
            Opcode::Fcmp => {
                let op = CmpOp::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                Operation::Fcmp { op, pd: rd, rs1, rs2 }
            }

            Opcode::Select => Operation::Select {
                rd,
                ps: rs1,
                rs1: rs2,
                rs2: modifier, // rs2 is encoded in modifier field for select
            },
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
                Operation::LocalLoad { width, rd, addr: rs1 }
            }
            Opcode::LocalStore => {
                let width = MemWidth::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
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
                Operation::DeviceLoad { width, rd, addr: rs1 }
            }
            Opcode::DeviceStore => {
                let width = MemWidth::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                    opcode,
                    modifier,
                    offset,
                })?;
                Operation::DeviceStore {
                    width,
                    addr: rs1,
                    value: rs2,
                }
            }

            Opcode::LocalAtomic => {
                if modifier == 8 && self.peek_u32().is_some() {
                    let word1 = self.read_u32();
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
                let scope_val =
                    Scope::from_u8(scope).ok_or(DecodeError::InvalidModifier {
                        opcode,
                        modifier: scope,
                        offset,
                    })?;

                if modifier == 8 && self.peek_u32().is_some() {
                    let word1 = self.read_u32();
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
                    let op =
                        WaveReduceType::from_u8(modifier - 8).ok_or(DecodeError::InvalidModifier {
                            opcode,
                            modifier,
                            offset,
                        })?;
                    Operation::WaveReduce { op, rd, rs1 }
                } else {
                    let op = WaveOpType::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                        opcode,
                        modifier,
                        offset,
                    })?;
                    match op {
                        WaveOpType::Ballot => Operation::WaveBallot { rd, ps: rs1 },
                        WaveOpType::Any | WaveOpType::All => {
                            Operation::WaveVote { op, pd: rd, ps: rs1 }
                        }
                        _ => Operation::WaveOp {
                            op,
                            rd,
                            rs1,
                            rs2: Some(rs2),
                        },
                    }
                }
            }

            Opcode::Control => self.decode_control(rd, rs1, rs2, modifier, scope, flags, offset)?,
        };

        Ok(op)
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_control(
        &mut self,
        rd: u8,
        rs1: u8,
        _rs2: u8,
        modifier: u8,
        scope: u8,
        flags: u8,
        offset: u32,
    ) -> Result<Operation, DecodeError> {
        if (flags & SYNC_OP_FLAG) != 0 {
            let op = SyncOp::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                opcode: Opcode::Control,
                modifier,
                offset,
            })?;
            match op {
                SyncOp::Return => Ok(Operation::Return),
                SyncOp::Halt => Ok(Operation::Halt),
                SyncOp::Barrier => Ok(Operation::Barrier),
                SyncOp::FenceAcquire => {
                    let scope_val = Scope::from_u8(scope).unwrap_or(Scope::Workgroup);
                    Ok(Operation::FenceAcquire { scope: scope_val })
                }
                SyncOp::FenceRelease => {
                    let scope_val = Scope::from_u8(scope).unwrap_or(Scope::Workgroup);
                    Ok(Operation::FenceRelease { scope: scope_val })
                }
                SyncOp::FenceAcqRel => {
                    let scope_val = Scope::from_u8(scope).unwrap_or(Scope::Workgroup);
                    Ok(Operation::FenceAcqRel { scope: scope_val })
                }
                SyncOp::Wait => Ok(Operation::Wait),
                SyncOp::Nop => Ok(Operation::Nop),
            }
        } else if (flags & MISC_OP_FLAG) != 0 {
            let op = MiscOp::from_u8(modifier).ok_or(DecodeError::InvalidModifier {
                opcode: Opcode::Control,
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

    fn encode_base(opcode: u8, rd: u8, rs1: u8, rs2: u8, modifier: u8, flags: u8) -> [u8; 4] {
        let word = ((u32::from(opcode) & OPCODE_MASK) << OPCODE_SHIFT)
            | ((u32::from(rd) & RD_MASK) << RD_SHIFT)
            | ((u32::from(rs1) & RS1_MASK) << RS1_SHIFT)
            | ((u32::from(rs2) & RS2_MASK) << RS2_SHIFT)
            | ((u32::from(modifier) & MODIFIER_MASK) << MODIFIER_SHIFT)
            | u32::from(flags);
        word.to_le_bytes()
    }

    #[test]
    fn test_decode_iadd() {
        let code = encode_base(0x00, 5, 3, 4, 0, 0);
        let instr = decode_at(&code, 0).unwrap();
        assert_eq!(instr.size, 4);
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
        let code = encode_base(0x3F, 0, 0, 0, SyncOp::Halt as u8, SYNC_OP_FLAG);
        let instr = decode_at(&code, 0).unwrap();
        assert_eq!(instr.operation, Operation::Halt);
    }

    #[test]
    fn test_decode_mov_imm() {
        let mut code = encode_base(0x3F, 5, 0, 0, MiscOp::MovImm as u8, MISC_OP_FLAG).to_vec();
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
        let code = encode_base(0x3F, 5, 4, 0, MiscOp::MovSr as u8, MISC_OP_FLAG);
        let instr = decode_at(&code, 0).unwrap();
        assert_eq!(
            instr.operation,
            Operation::MovSr {
                rd: 5,
                sr_index: 4
            }
        );
    }

    #[test]
    fn test_decode_device_store_u32() {
        let code = encode_base(0x39, 0, 3, 5, MemWidth::U32 as u8, 0);
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
        let mut code = encode_base(0x00, 5, 3, 4, 0, 0).to_vec(); // iadd
        code.extend_from_slice(&encode_base(0x3F, 0, 0, 0, SyncOp::Halt as u8, SYNC_OP_FLAG)); // halt

        let instructions = decode_all(&code).unwrap();
        assert_eq!(instructions.len(), 2);
        assert_eq!(instructions[0].offset, 0);
        assert_eq!(instructions[1].offset, 4);
    }

    #[test]
    fn test_decode_shl() {
        let code = encode_base(0x24, 2, 0, 1, 0, 0);
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
    fn test_decode_predicated() {
        let word = ((0x00u32) << OPCODE_SHIFT)
            | ((5u32) << RD_SHIFT)
            | ((3u32) << RS1_SHIFT)
            | ((4u32) << RS2_SHIFT)
            | ((1u32) << PRED_SHIFT); // p1

        let code = word.to_le_bytes();
        let instr = decode_at(&code, 0).unwrap();
        assert_eq!(instr.predicate, 1);
        assert!(!instr.predicate_negated);
    }
}
