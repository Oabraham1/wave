// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! LIR instruction definitions mapping 1:1 to WAVE ISA instructions.
//!
//! LIR uses virtual registers (VReg) and predicate registers (PReg).
//! After register allocation, VRegs are replaced with physical registers.

use super::operand::{MemWidth, PReg, SpecialReg, VReg};

/// A near-WAVE instruction with virtual registers.
#[derive(Debug, Clone, PartialEq)]
pub enum LirInst {
    /// Integer addition.
    Iadd { dest: VReg, src1: VReg, src2: VReg },
    /// Integer subtraction.
    Isub { dest: VReg, src1: VReg, src2: VReg },
    /// Integer multiplication.
    Imul { dest: VReg, src1: VReg, src2: VReg },
    /// Integer division.
    Idiv { dest: VReg, src1: VReg, src2: VReg },
    /// Integer modulo.
    Imod { dest: VReg, src1: VReg, src2: VReg },
    /// Integer negation.
    Ineg { dest: VReg, src: VReg },
    /// Float addition.
    Fadd { dest: VReg, src1: VReg, src2: VReg },
    /// Float subtraction.
    Fsub { dest: VReg, src1: VReg, src2: VReg },
    /// Float multiplication.
    Fmul { dest: VReg, src1: VReg, src2: VReg },
    /// Float division.
    Fdiv { dest: VReg, src1: VReg, src2: VReg },
    /// Fused multiply-add: dest = src1 * src2 + src3.
    Fma { dest: VReg, src1: VReg, src2: VReg, src3: VReg },
    /// Float negation.
    Fneg { dest: VReg, src: VReg },
    /// Float absolute value.
    Fabs { dest: VReg, src: VReg },
    /// Float square root.
    Fsqrt { dest: VReg, src: VReg },
    /// Float sine.
    Fsin { dest: VReg, src: VReg },
    /// Float cosine.
    Fcos { dest: VReg, src: VReg },
    /// Float base-2 exponential.
    Fexp2 { dest: VReg, src: VReg },
    /// Float base-2 logarithm.
    Flog2 { dest: VReg, src: VReg },
    /// Float minimum.
    Fmin { dest: VReg, src1: VReg, src2: VReg },
    /// Float maximum.
    Fmax { dest: VReg, src1: VReg, src2: VReg },
    /// Move immediate value to register.
    MovImm { dest: VReg, value: u32 },
    /// Move register to register.
    MovReg { dest: VReg, src: VReg },
    /// Move special register to register.
    MovSr { dest: VReg, sr: SpecialReg },
    /// Bitwise AND.
    And { dest: VReg, src1: VReg, src2: VReg },
    /// Bitwise OR.
    Or { dest: VReg, src1: VReg, src2: VReg },
    /// Bitwise XOR.
    Xor { dest: VReg, src1: VReg, src2: VReg },
    /// Bitwise NOT.
    Not { dest: VReg, src: VReg },
    /// Shift left.
    Shl { dest: VReg, src1: VReg, src2: VReg },
    /// Shift right (logical).
    Shr { dest: VReg, src1: VReg, src2: VReg },
    /// Shift right (arithmetic).
    Sar { dest: VReg, src1: VReg, src2: VReg },
    /// Load from local (shared) memory.
    LocalLoad { dest: VReg, addr: VReg, width: MemWidth },
    /// Store to local (shared) memory.
    LocalStore { addr: VReg, value: VReg, width: MemWidth },
    /// Load from device (global) memory.
    DeviceLoad { dest: VReg, addr: VReg, width: MemWidth },
    /// Store to device (global) memory.
    DeviceStore { addr: VReg, value: VReg, width: MemWidth },
    /// Signed integer compare equal.
    IcmpEq { dest: PReg, src1: VReg, src2: VReg },
    /// Signed integer compare not equal.
    IcmpNe { dest: PReg, src1: VReg, src2: VReg },
    /// Signed integer compare less than.
    IcmpLt { dest: PReg, src1: VReg, src2: VReg },
    /// Signed integer compare less or equal.
    IcmpLe { dest: PReg, src1: VReg, src2: VReg },
    /// Signed integer compare greater than.
    IcmpGt { dest: PReg, src1: VReg, src2: VReg },
    /// Signed integer compare greater or equal.
    IcmpGe { dest: PReg, src1: VReg, src2: VReg },
    /// Unsigned integer compare less than.
    UcmpLt { dest: PReg, src1: VReg, src2: VReg },
    /// Float compare equal.
    FcmpEq { dest: PReg, src1: VReg, src2: VReg },
    /// Float compare less than.
    FcmpLt { dest: PReg, src1: VReg, src2: VReg },
    /// Float compare greater than.
    FcmpGt { dest: PReg, src1: VReg, src2: VReg },
    /// Convert float to int.
    CvtF32I32 { dest: VReg, src: VReg },
    /// Convert int to float.
    CvtI32F32 { dest: VReg, src: VReg },
    /// Conditional (structured if).
    If { pred: PReg },
    /// Else branch.
    Else,
    /// End conditional.
    Endif,
    /// Loop start.
    Loop,
    /// Break out of loop.
    Break { pred: PReg },
    /// Continue to next iteration.
    Continue { pred: PReg },
    /// End loop.
    Endloop,
    /// Workgroup barrier.
    Barrier,
    /// Halt execution.
    Halt,
}

impl LirInst {
    /// Returns the destination VReg if this instruction defines one.
    #[must_use]
    pub fn dest_vreg(&self) -> Option<VReg> {
        match self {
            Self::Iadd { dest, .. }
            | Self::Isub { dest, .. }
            | Self::Imul { dest, .. }
            | Self::Idiv { dest, .. }
            | Self::Imod { dest, .. }
            | Self::Ineg { dest, .. }
            | Self::Fadd { dest, .. }
            | Self::Fsub { dest, .. }
            | Self::Fmul { dest, .. }
            | Self::Fdiv { dest, .. }
            | Self::Fma { dest, .. }
            | Self::Fneg { dest, .. }
            | Self::Fabs { dest, .. }
            | Self::Fsqrt { dest, .. }
            | Self::Fsin { dest, .. }
            | Self::Fcos { dest, .. }
            | Self::Fexp2 { dest, .. }
            | Self::Flog2 { dest, .. }
            | Self::Fmin { dest, .. }
            | Self::Fmax { dest, .. }
            | Self::MovImm { dest, .. }
            | Self::MovReg { dest, .. }
            | Self::MovSr { dest, .. }
            | Self::And { dest, .. }
            | Self::Or { dest, .. }
            | Self::Xor { dest, .. }
            | Self::Not { dest, .. }
            | Self::Shl { dest, .. }
            | Self::Shr { dest, .. }
            | Self::Sar { dest, .. }
            | Self::LocalLoad { dest, .. }
            | Self::DeviceLoad { dest, .. }
            | Self::CvtF32I32 { dest, .. }
            | Self::CvtI32F32 { dest, .. } => Some(*dest),
            _ => None,
        }
    }

    /// Returns all VRegs used as source operands.
    #[must_use]
    pub fn src_vregs(&self) -> Vec<VReg> {
        match self {
            Self::Iadd { src1, src2, .. }
            | Self::Isub { src1, src2, .. }
            | Self::Imul { src1, src2, .. }
            | Self::Idiv { src1, src2, .. }
            | Self::Imod { src1, src2, .. }
            | Self::Fadd { src1, src2, .. }
            | Self::Fsub { src1, src2, .. }
            | Self::Fmul { src1, src2, .. }
            | Self::Fdiv { src1, src2, .. }
            | Self::Fmin { src1, src2, .. }
            | Self::Fmax { src1, src2, .. }
            | Self::And { src1, src2, .. }
            | Self::Or { src1, src2, .. }
            | Self::Xor { src1, src2, .. }
            | Self::Shl { src1, src2, .. }
            | Self::Shr { src1, src2, .. }
            | Self::Sar { src1, src2, .. } => vec![*src1, *src2],
            Self::Fma {
                src1, src2, src3, ..
            } => vec![*src1, *src2, *src3],
            Self::Ineg { src, .. }
            | Self::Fneg { src, .. }
            | Self::Fabs { src, .. }
            | Self::Fsqrt { src, .. }
            | Self::Fsin { src, .. }
            | Self::Fcos { src, .. }
            | Self::Fexp2 { src, .. }
            | Self::Flog2 { src, .. }
            | Self::Not { src, .. }
            | Self::MovReg { src, .. }
            | Self::CvtF32I32 { src, .. }
            | Self::CvtI32F32 { src, .. } => vec![*src],
            Self::LocalLoad { addr, .. } | Self::DeviceLoad { addr, .. } => vec![*addr],
            Self::LocalStore { addr, value, .. } | Self::DeviceStore { addr, value, .. } => {
                vec![*addr, *value]
            }
            Self::IcmpEq { src1, src2, .. }
            | Self::IcmpNe { src1, src2, .. }
            | Self::IcmpLt { src1, src2, .. }
            | Self::IcmpLe { src1, src2, .. }
            | Self::IcmpGt { src1, src2, .. }
            | Self::IcmpGe { src1, src2, .. }
            | Self::UcmpLt { src1, src2, .. }
            | Self::FcmpEq { src1, src2, .. }
            | Self::FcmpLt { src1, src2, .. }
            | Self::FcmpGt { src1, src2, .. } => vec![*src1, *src2],
            Self::MovImm { .. }
            | Self::MovSr { .. }
            | Self::If { .. }
            | Self::Else
            | Self::Endif
            | Self::Loop
            | Self::Break { .. }
            | Self::Continue { .. }
            | Self::Endloop
            | Self::Barrier
            | Self::Halt => vec![],
        }
    }

    /// Returns true if this is a move instruction (used for coalescing).
    #[must_use]
    pub fn is_move(&self) -> bool {
        matches!(self, Self::MovReg { .. })
    }

    /// Returns true if this instruction has side effects.
    #[must_use]
    pub fn has_side_effects(&self) -> bool {
        matches!(
            self,
            Self::LocalStore { .. }
                | Self::DeviceStore { .. }
                | Self::Barrier
                | Self::Halt
                | Self::If { .. }
                | Self::Else
                | Self::Endif
                | Self::Loop
                | Self::Break { .. }
                | Self::Continue { .. }
                | Self::Endloop
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dest_and_src_vregs() {
        let inst = LirInst::Iadd {
            dest: VReg(2),
            src1: VReg(0),
            src2: VReg(1),
        };
        assert_eq!(inst.dest_vreg(), Some(VReg(2)));
        assert_eq!(inst.src_vregs(), vec![VReg(0), VReg(1)]);
    }

    #[test]
    fn test_store_no_dest() {
        let inst = LirInst::DeviceStore {
            addr: VReg(0),
            value: VReg(1),
            width: MemWidth::W32,
        };
        assert_eq!(inst.dest_vreg(), None);
        assert_eq!(inst.src_vregs(), vec![VReg(0), VReg(1)]);
        assert!(inst.has_side_effects());
    }

    #[test]
    fn test_move_classification() {
        let mov = LirInst::MovReg {
            dest: VReg(1),
            src: VReg(0),
        };
        assert!(mov.is_move());
        let add = LirInst::Iadd {
            dest: VReg(2),
            src1: VReg(0),
            src2: VReg(1),
        };
        assert!(!add.is_move());
    }

    #[test]
    fn test_fma_three_sources() {
        let inst = LirInst::Fma {
            dest: VReg(3),
            src1: VReg(0),
            src2: VReg(1),
            src3: VReg(2),
        };
        assert_eq!(inst.dest_vreg(), Some(VReg(3)));
        assert_eq!(inst.src_vregs(), vec![VReg(0), VReg(1), VReg(2)]);
    }
}
