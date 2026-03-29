// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Pretty-printing for LIR instructions.
//!
//! Formats LIR instructions in a readable assembly-like syntax
//! for debugging and compiler development.

use std::fmt::Write;

use super::instruction::LirInst;

/// Format a single LIR instruction as a string.
#[must_use]
pub fn format_lir_inst(inst: &LirInst) -> String {
    match inst {
        LirInst::Iadd { dest, src1, src2 } => format!("iadd {dest}, {src1}, {src2}"),
        LirInst::Isub { dest, src1, src2 } => format!("isub {dest}, {src1}, {src2}"),
        LirInst::Imul { dest, src1, src2 } => format!("imul {dest}, {src1}, {src2}"),
        LirInst::Idiv { dest, src1, src2 } => format!("idiv {dest}, {src1}, {src2}"),
        LirInst::Imod { dest, src1, src2 } => format!("imod {dest}, {src1}, {src2}"),
        LirInst::Ineg { dest, src } => format!("ineg {dest}, {src}"),
        LirInst::Fadd { dest, src1, src2 } => format!("fadd {dest}, {src1}, {src2}"),
        LirInst::Fsub { dest, src1, src2 } => format!("fsub {dest}, {src1}, {src2}"),
        LirInst::Fmul { dest, src1, src2 } => format!("fmul {dest}, {src1}, {src2}"),
        LirInst::Fdiv { dest, src1, src2 } => format!("fdiv {dest}, {src1}, {src2}"),
        LirInst::Fma {
            dest,
            src1,
            src2,
            src3,
        } => {
            format!("fma {dest}, {src1}, {src2}, {src3}")
        }
        LirInst::Fneg { dest, src } => format!("fneg {dest}, {src}"),
        LirInst::Fabs { dest, src } => format!("fabs {dest}, {src}"),
        LirInst::Fsqrt { dest, src } => format!("fsqrt {dest}, {src}"),
        LirInst::Fsin { dest, src } => format!("fsin {dest}, {src}"),
        LirInst::Fcos { dest, src } => format!("fcos {dest}, {src}"),
        LirInst::Fexp2 { dest, src } => format!("fexp2 {dest}, {src}"),
        LirInst::Flog2 { dest, src } => format!("flog2 {dest}, {src}"),
        LirInst::Fmin { dest, src1, src2 } => format!("fmin {dest}, {src1}, {src2}"),
        LirInst::Fmax { dest, src1, src2 } => format!("fmax {dest}, {src1}, {src2}"),
        LirInst::MovImm { dest, value } => format!("mov_imm {dest}, 0x{value:08x}"),
        LirInst::MovReg { dest, src } => format!("mov {dest}, {src}"),
        LirInst::MovSr { dest, sr } => format!("mov {dest}, sr{}", sr.index()),
        LirInst::And { dest, src1, src2 } => format!("and {dest}, {src1}, {src2}"),
        LirInst::Or { dest, src1, src2 } => format!("or {dest}, {src1}, {src2}"),
        LirInst::Xor { dest, src1, src2 } => format!("xor {dest}, {src1}, {src2}"),
        LirInst::Not { dest, src } => format!("not {dest}, {src}"),
        LirInst::Shl { dest, src1, src2 } => format!("shl {dest}, {src1}, {src2}"),
        LirInst::Shr { dest, src1, src2 } => format!("shr {dest}, {src1}, {src2}"),
        LirInst::Sar { dest, src1, src2 } => format!("sar {dest}, {src1}, {src2}"),
        LirInst::LocalLoad { dest, addr, width } => {
            format!("local_load_{} {dest}, [{addr}]", width_suffix(*width))
        }
        LirInst::LocalStore { addr, value, width } => {
            format!("local_store_{} [{addr}], {value}", width_suffix(*width))
        }
        LirInst::DeviceLoad { dest, addr, width } => {
            format!("device_load_{} {dest}, [{addr}]", width_suffix(*width))
        }
        LirInst::DeviceStore { addr, value, width } => {
            format!("device_store_{} [{addr}], {value}", width_suffix(*width))
        }
        LirInst::IcmpEq { dest, src1, src2 } => format!("icmp_eq {dest}, {src1}, {src2}"),
        LirInst::IcmpNe { dest, src1, src2 } => format!("icmp_ne {dest}, {src1}, {src2}"),
        LirInst::IcmpLt { dest, src1, src2 } => format!("icmp_lt {dest}, {src1}, {src2}"),
        LirInst::IcmpLe { dest, src1, src2 } => format!("icmp_le {dest}, {src1}, {src2}"),
        LirInst::IcmpGt { dest, src1, src2 } => format!("icmp_gt {dest}, {src1}, {src2}"),
        LirInst::IcmpGe { dest, src1, src2 } => format!("icmp_ge {dest}, {src1}, {src2}"),
        LirInst::UcmpLt { dest, src1, src2 } => format!("ucmp_lt {dest}, {src1}, {src2}"),
        LirInst::FcmpEq { dest, src1, src2 } => format!("fcmp_eq {dest}, {src1}, {src2}"),
        LirInst::FcmpLt { dest, src1, src2 } => format!("fcmp_lt {dest}, {src1}, {src2}"),
        LirInst::FcmpGt { dest, src1, src2 } => format!("fcmp_gt {dest}, {src1}, {src2}"),
        LirInst::CvtF32I32 { dest, src } => format!("cvt_f32_i32 {dest}, {src}"),
        LirInst::CvtI32F32 { dest, src } => format!("cvt_i32_f32 {dest}, {src}"),
        LirInst::If { pred } => format!("if {pred}"),
        LirInst::Else => "else".to_string(),
        LirInst::Endif => "endif".to_string(),
        LirInst::Loop => "loop".to_string(),
        LirInst::Break { pred } => format!("break {pred}"),
        LirInst::Continue { pred } => format!("continue {pred}"),
        LirInst::Endloop => "endloop".to_string(),
        LirInst::Barrier => "barrier".to_string(),
        LirInst::Halt => "halt".to_string(),
    }
}

fn width_suffix(w: super::operand::MemWidth) -> &'static str {
    match w {
        super::operand::MemWidth::W8 => "u8",
        super::operand::MemWidth::W16 => "u16",
        super::operand::MemWidth::W32 => "u32",
        super::operand::MemWidth::W64 => "u64",
    }
}

/// Format a list of LIR instructions.
#[must_use]
pub fn display_lir(instructions: &[LirInst]) -> String {
    let mut out = String::new();
    for inst in instructions {
        let _ = writeln!(out, "  {}", format_lir_inst(inst));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lir::operand::{MemWidth, PReg, SpecialReg, VReg};

    #[test]
    fn test_format_arithmetic() {
        let inst = LirInst::Iadd {
            dest: VReg(2),
            src1: VReg(0),
            src2: VReg(1),
        };
        assert_eq!(format_lir_inst(&inst), "iadd v2, v0, v1");
    }

    #[test]
    fn test_format_memory_ops() {
        let load = LirInst::DeviceLoad {
            dest: VReg(1),
            addr: VReg(0),
            width: MemWidth::W32,
        };
        assert_eq!(format_lir_inst(&load), "device_load_u32 v1, [v0]");

        let store = LirInst::DeviceStore {
            addr: VReg(0),
            value: VReg(1),
            width: MemWidth::W32,
        };
        assert_eq!(format_lir_inst(&store), "device_store_u32 [v0], v1");
    }

    #[test]
    fn test_format_control_flow() {
        assert_eq!(format_lir_inst(&LirInst::If { pred: PReg(0) }), "if p0");
        assert_eq!(format_lir_inst(&LirInst::Endif), "endif");
        assert_eq!(format_lir_inst(&LirInst::Halt), "halt");
    }

    #[test]
    fn test_format_special_reg() {
        let inst = LirInst::MovSr {
            dest: VReg(0),
            sr: SpecialReg::ThreadIdX,
        };
        assert_eq!(format_lir_inst(&inst), "mov v0, sr0");
    }
}
