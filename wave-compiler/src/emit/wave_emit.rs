// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! LIR to WAVE instruction emission.
//!
//! Converts LIR instructions with physical registers to WAVE encoded
//! instruction words. Uses wave-decode's opcode and field definitions
//! for bit-accurate encoding.

use wave_decode::opcodes::{
    CmpOp, ControlOp, FUnaryOp, MemWidth as DecodeMemWidth, MiscOp, Opcode, SyncOp, MODIFIER_MASK,
    MODIFIER_SHIFT, OPCODE_SHIFT, RD_SHIFT, RS1_SHIFT, RS2_SHIFT,
};

use crate::lir::instruction::LirInst;
use crate::lir::operand::{MemWidth, PhysReg, VReg};

const SYNC_OP_FLAG: u32 = 0x01;
const MISC_OP_FLAG: u32 = 0x02;

/// An encoded WAVE instruction (4 or 8 bytes).
#[derive(Debug, Clone)]
pub struct EncodedInst {
    /// First instruction word.
    pub word0: u32,
    /// Optional second word for extended instructions.
    pub word1: Option<u32>,
}

impl EncodedInst {
    /// Returns the size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        if self.word1.is_some() {
            8
        } else {
            4
        }
    }

    /// Encode to bytes (little-endian).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = self.word0.to_le_bytes().to_vec();
        if let Some(w1) = self.word1 {
            bytes.extend_from_slice(&w1.to_le_bytes());
        }
        bytes
    }
}

/// Mapping from `VReg` to physical register index.
pub type RegMap = std::collections::HashMap<VReg, PhysReg>;

/// Emit a LIR instruction as encoded WAVE bytes using the register mapping.
///
/// For instructions that still use `VReg`s (before register allocation),
/// the `VReg` number is used directly as the physical register index.
#[must_use]
pub fn emit_instruction(inst: &LirInst, reg_map: &RegMap) -> EncodedInst {
    match inst {
        LirInst::Iadd { dest, src1, src2 } => encode_alu3(
            Opcode::Iadd,
            reg(dest, reg_map),
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::Isub { dest, src1, src2 } => encode_alu3(
            Opcode::Isub,
            reg(dest, reg_map),
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::Imul { dest, src1, src2 } => encode_alu3(
            Opcode::Imul,
            reg(dest, reg_map),
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::Idiv { dest, src1, src2 } => encode_alu3(
            Opcode::Idiv,
            reg(dest, reg_map),
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::Imod { dest, src1, src2 } => encode_alu3(
            Opcode::Imod,
            reg(dest, reg_map),
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::Ineg { dest, src } => {
            encode_alu2(Opcode::Ineg, reg(dest, reg_map), reg(src, reg_map))
        }
        LirInst::Fadd { dest, src1, src2 } => encode_alu3(
            Opcode::Fadd,
            reg(dest, reg_map),
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::Fsub { dest, src1, src2 } => encode_alu3(
            Opcode::Fsub,
            reg(dest, reg_map),
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::Fmul { dest, src1, src2 } => encode_alu3(
            Opcode::Fmul,
            reg(dest, reg_map),
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::Fdiv { dest, src1, src2 } => encode_alu3(
            Opcode::Fdiv,
            reg(dest, reg_map),
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::Fma {
            dest,
            src1,
            src2,
            src3,
        } => {
            let word0 = encode_base_word(
                Opcode::Fma,
                reg(dest, reg_map),
                reg(src1, reg_map),
                reg(src2, reg_map),
                0,
            );
            let word1 = u32::from(reg(src3, reg_map)) << 27;
            EncodedInst {
                word0,
                word1: Some(word1),
            }
        }
        LirInst::Fneg { dest, src } => {
            encode_alu2(Opcode::Fneg, reg(dest, reg_map), reg(src, reg_map))
        }
        LirInst::Fabs { dest, src } => {
            encode_alu2(Opcode::Fabs, reg(dest, reg_map), reg(src, reg_map))
        }
        LirInst::Fsqrt { dest, src } => {
            encode_alu2(Opcode::Fsqrt, reg(dest, reg_map), reg(src, reg_map))
        }
        LirInst::Fsin { dest, src } => {
            encode_funary(FUnaryOp::Fsin, reg(dest, reg_map), reg(src, reg_map))
        }
        LirInst::Fcos { dest, src } => {
            encode_funary(FUnaryOp::Fcos, reg(dest, reg_map), reg(src, reg_map))
        }
        LirInst::Fexp2 { dest, src } => {
            encode_funary(FUnaryOp::Fexp2, reg(dest, reg_map), reg(src, reg_map))
        }
        LirInst::Flog2 { dest, src } => {
            encode_funary(FUnaryOp::Flog2, reg(dest, reg_map), reg(src, reg_map))
        }
        LirInst::Fmin { dest, src1, src2 } => encode_alu3(
            Opcode::Fmin,
            reg(dest, reg_map),
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::Fmax { dest, src1, src2 } => encode_alu3(
            Opcode::Fmax,
            reg(dest, reg_map),
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::MovImm { dest, value } => {
            let word0 =
                encode_control_word(MiscOp::MovImm as u8, reg(dest, reg_map), 0, 0) | MISC_OP_FLAG;
            EncodedInst {
                word0,
                word1: Some(*value),
            }
        }
        LirInst::MovReg { dest, src } => {
            let word0 =
                encode_control_word(MiscOp::Mov as u8, reg(dest, reg_map), reg(src, reg_map), 0)
                    | MISC_OP_FLAG;
            EncodedInst { word0, word1: None }
        }
        LirInst::MovSr { dest, sr } => {
            let word0 = encode_control_word(MiscOp::MovSr as u8, reg(dest, reg_map), sr.index(), 0)
                | MISC_OP_FLAG;
            EncodedInst { word0, word1: None }
        }
        LirInst::And { dest, src1, src2 } => encode_alu3(
            Opcode::And,
            reg(dest, reg_map),
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::Or { dest, src1, src2 } => encode_alu3(
            Opcode::Or,
            reg(dest, reg_map),
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::Xor { dest, src1, src2 } => encode_alu3(
            Opcode::Xor,
            reg(dest, reg_map),
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::Not { dest, src } => {
            encode_alu2(Opcode::Not, reg(dest, reg_map), reg(src, reg_map))
        }
        LirInst::Shl { dest, src1, src2 } => encode_alu3(
            Opcode::Shl,
            reg(dest, reg_map),
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::Shr { dest, src1, src2 } => encode_alu3(
            Opcode::Shr,
            reg(dest, reg_map),
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::Sar { dest, src1, src2 } => encode_alu3(
            Opcode::Sar,
            reg(dest, reg_map),
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::LocalLoad { dest, addr, width } => encode_mem(
            Opcode::LocalLoad,
            reg(dest, reg_map),
            reg(addr, reg_map),
            0,
            lir_width_to_decode(*width),
        ),
        LirInst::LocalStore { addr, value, width } => encode_mem(
            Opcode::LocalStore,
            0,
            reg(addr, reg_map),
            reg(value, reg_map),
            lir_width_to_decode(*width),
        ),
        LirInst::DeviceLoad { dest, addr, width } => encode_mem(
            Opcode::DeviceLoad,
            reg(dest, reg_map),
            reg(addr, reg_map),
            0,
            lir_width_to_decode(*width),
        ),
        LirInst::DeviceStore { addr, value, width } => encode_mem(
            Opcode::DeviceStore,
            0,
            reg(addr, reg_map),
            reg(value, reg_map),
            lir_width_to_decode(*width),
        ),
        LirInst::IcmpEq { dest, src1, src2 } => encode_cmp(
            Opcode::Icmp,
            CmpOp::Eq,
            dest.0,
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::IcmpNe { dest, src1, src2 } => encode_cmp(
            Opcode::Icmp,
            CmpOp::Ne,
            dest.0,
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::IcmpLt { dest, src1, src2 } => encode_cmp(
            Opcode::Icmp,
            CmpOp::Lt,
            dest.0,
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::IcmpLe { dest, src1, src2 } => encode_cmp(
            Opcode::Icmp,
            CmpOp::Le,
            dest.0,
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::IcmpGt { dest, src1, src2 } => encode_cmp(
            Opcode::Icmp,
            CmpOp::Gt,
            dest.0,
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::IcmpGe { dest, src1, src2 } => encode_cmp(
            Opcode::Icmp,
            CmpOp::Ge,
            dest.0,
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::UcmpLt { dest, src1, src2 } => encode_cmp(
            Opcode::Ucmp,
            CmpOp::Lt,
            dest.0,
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::FcmpEq { dest, src1, src2 } => encode_cmp(
            Opcode::Fcmp,
            CmpOp::Eq,
            dest.0,
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::FcmpLt { dest, src1, src2 } => encode_cmp(
            Opcode::Fcmp,
            CmpOp::Lt,
            dest.0,
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::FcmpGt { dest, src1, src2 } => encode_cmp(
            Opcode::Fcmp,
            CmpOp::Gt,
            dest.0,
            reg(src1, reg_map),
            reg(src2, reg_map),
        ),
        LirInst::CvtF32I32 { dest, src } => {
            let word0 = encode_base_word(Opcode::Cvt, reg(dest, reg_map), reg(src, reg_map), 0, 0);
            EncodedInst { word0, word1: None }
        }
        LirInst::CvtI32F32 { dest, src } => {
            let word0 = encode_base_word(Opcode::Cvt, reg(dest, reg_map), reg(src, reg_map), 0, 2);
            EncodedInst { word0, word1: None }
        }
        LirInst::If { pred } => {
            let word0 = encode_control_word(ControlOp::If as u8, 0, pred.0, 0);
            EncodedInst { word0, word1: None }
        }
        LirInst::Else => {
            let word0 = encode_control_word(ControlOp::Else as u8, 0, 0, 0);
            EncodedInst { word0, word1: None }
        }
        LirInst::Endif => {
            let word0 = encode_control_word(ControlOp::Endif as u8, 0, 0, 0);
            EncodedInst { word0, word1: None }
        }
        LirInst::Loop => {
            let word0 = encode_control_word(ControlOp::Loop as u8, 0, 0, 0);
            EncodedInst { word0, word1: None }
        }
        LirInst::Break { pred } => {
            let word0 = encode_control_word(ControlOp::Break as u8, 0, pred.0, 0);
            EncodedInst { word0, word1: None }
        }
        LirInst::Continue { pred } => {
            let word0 = encode_control_word(ControlOp::Continue as u8, 0, pred.0, 0);
            EncodedInst { word0, word1: None }
        }
        LirInst::Endloop => {
            let word0 = encode_control_word(ControlOp::Endloop as u8, 0, 0, 0);
            EncodedInst { word0, word1: None }
        }
        LirInst::Barrier => {
            let word0 = encode_control_word(SyncOp::Barrier as u8, 0, 0, 0) | SYNC_OP_FLAG;
            EncodedInst { word0, word1: None }
        }
        LirInst::Halt => {
            let word0 = encode_control_word(SyncOp::Halt as u8, 0, 0, 0) | SYNC_OP_FLAG;
            EncodedInst { word0, word1: None }
        }
    }
}

fn reg(vreg: &VReg, map: &RegMap) -> u8 {
    map.get(vreg).map_or(vreg.0 as u8, |p| p.0)
}

fn encode_base_word(opcode: Opcode, rd: u8, rs1: u8, rs2: u8, modifier: u8) -> u32 {
    (u32::from(opcode as u8) << OPCODE_SHIFT)
        | (u32::from(rd & 0x1F) << RD_SHIFT)
        | (u32::from(rs1 & 0x1F) << RS1_SHIFT)
        | (u32::from(rs2 & 0x1F) << RS2_SHIFT)
        | ((u32::from(modifier) & MODIFIER_MASK) << MODIFIER_SHIFT)
}

fn encode_alu3(opcode: Opcode, rd: u8, rs1: u8, rs2: u8) -> EncodedInst {
    EncodedInst {
        word0: encode_base_word(opcode, rd, rs1, rs2, 0),
        word1: None,
    }
}

fn encode_alu2(opcode: Opcode, rd: u8, rs1: u8) -> EncodedInst {
    EncodedInst {
        word0: encode_base_word(opcode, rd, rs1, 0, 0),
        word1: None,
    }
}

fn encode_funary(op: FUnaryOp, rd: u8, rs1: u8) -> EncodedInst {
    EncodedInst {
        word0: encode_base_word(Opcode::FUnaryOps, rd, rs1, 0, op as u8),
        word1: None,
    }
}

fn encode_mem(opcode: Opcode, rd: u8, rs1: u8, rs2: u8, width: DecodeMemWidth) -> EncodedInst {
    EncodedInst {
        word0: encode_base_word(opcode, rd, rs1, rs2, width as u8),
        word1: None,
    }
}

fn encode_cmp(opcode: Opcode, cmp_op: CmpOp, pd: u8, rs1: u8, rs2: u8) -> EncodedInst {
    EncodedInst {
        word0: encode_base_word(opcode, pd, rs1, rs2, cmp_op as u8),
        word1: None,
    }
}

fn encode_control_word(modifier: u8, rd: u8, rs1: u8, rs2: u8) -> u32 {
    encode_base_word(Opcode::Control, rd, rs1, rs2, modifier)
}

fn lir_width_to_decode(w: MemWidth) -> DecodeMemWidth {
    match w {
        MemWidth::W8 => DecodeMemWidth::U8,
        MemWidth::W16 => DecodeMemWidth::U16,
        MemWidth::W32 => DecodeMemWidth::U32,
        MemWidth::W64 => DecodeMemWidth::U64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn empty_map() -> RegMap {
        HashMap::new()
    }

    #[test]
    fn test_emit_iadd() {
        let inst = LirInst::Iadd {
            dest: VReg(5),
            src1: VReg(3),
            src2: VReg(4),
        };
        let encoded = emit_instruction(&inst, &empty_map());
        assert!(encoded.word1.is_none());
        let opcode = (encoded.word0 >> OPCODE_SHIFT) & 0x3F;
        assert_eq!(opcode, Opcode::Iadd as u32);
        let rd = (encoded.word0 >> RD_SHIFT) & 0x1F;
        assert_eq!(rd, 5);
    }

    #[test]
    fn test_emit_device_store_encoding() {
        let inst = LirInst::DeviceStore {
            addr: VReg(3),
            value: VReg(5),
            width: MemWidth::W32,
        };
        let encoded = emit_instruction(&inst, &empty_map());
        let rs1 = (encoded.word0 >> RS1_SHIFT) & 0x1F;
        let rs2 = (encoded.word0 >> RS2_SHIFT) & 0x1F;
        assert_eq!(rs1, 3);
        assert_eq!(rs2, 5);
    }

    #[test]
    fn test_emit_mov_imm() {
        let inst = LirInst::MovImm {
            dest: VReg(5),
            value: 0x1234_5678,
        };
        let encoded = emit_instruction(&inst, &empty_map());
        assert!(encoded.word1.is_some());
        assert_eq!(encoded.word1.unwrap(), 0x1234_5678);
    }

    #[test]
    fn test_emit_halt() {
        let inst = LirInst::Halt;
        let encoded = emit_instruction(&inst, &empty_map());
        let opcode = (encoded.word0 >> OPCODE_SHIFT) & 0x3F;
        assert_eq!(opcode, Opcode::Control as u32);
        assert_ne!(encoded.word0 & SYNC_OP_FLAG, 0);
    }

    #[test]
    fn test_emit_if_endif() {
        let if_inst = LirInst::If {
            pred: crate::lir::operand::PReg(1),
        };
        let encoded_if = emit_instruction(&if_inst, &empty_map());
        let modifier = (encoded_if.word0 >> MODIFIER_SHIFT) & 0x0F;
        assert_eq!(modifier, ControlOp::If as u32);

        let endif_inst = LirInst::Endif;
        let encoded_endif = emit_instruction(&endif_inst, &empty_map());
        let modifier = (encoded_endif.word0 >> MODIFIER_SHIFT) & 0x0F;
        assert_eq!(modifier, ControlOp::Endif as u32);
    }

    #[test]
    fn test_emit_with_reg_map() {
        let mut map = RegMap::new();
        map.insert(VReg(0), PhysReg(10));
        map.insert(VReg(1), PhysReg(11));
        map.insert(VReg(2), PhysReg(12));

        let inst = LirInst::Iadd {
            dest: VReg(2),
            src1: VReg(0),
            src2: VReg(1),
        };
        let encoded = emit_instruction(&inst, &map);
        let rd = (encoded.word0 >> RD_SHIFT) & 0x1F;
        let rs1 = (encoded.word0 >> RS1_SHIFT) & 0x1F;
        let rs2 = (encoded.word0 >> RS2_SHIFT) & 0x1F;
        assert_eq!(rd, 12);
        assert_eq!(rs1, 10);
        assert_eq!(rs2, 11);
    }
}
