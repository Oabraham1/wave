// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Main code generator. Iterates decoded WAVE instructions and emits equivalent PTX
//!
//! assembly text. Integer operations map directly to PTX .s32/.u32 instructions. Float
//! operations require mov.b32 bitcasting between %r (b32) and %f (f32) register sets
//! since WAVE registers are untyped. Control flow lowers structured if/loop constructs
//! to predicated branches with generated labels. Predicated WAVE instructions emit the
//! PTX @%p / @!%p prefix on each generated instruction line.

use std::fmt::Write;

use wave_decode::opcodes::{
    BitOpType, CmpOp, CvtType, F16Op, F16PackedOp, F64DivSqrtOp, F64Op, FUnaryOp,
};
use wave_decode::{DecodedInstruction, KernelInfo, Operation};

use crate::control_flow::ControlFlowState;
use crate::intrinsics;
use crate::memory;
use crate::registers::{self, freg, pred, reg};
use crate::CompileError;

pub struct CodeGenerator {
    output: String,
    indent_level: usize,
    cf: ControlFlowState,
}

impl Default for CodeGenerator {
    fn default() -> Self {
        Self {
            output: String::new(),
            indent_level: 1,
            cf: ControlFlowState::new(),
        }
    }
}

#[allow(clippy::similar_names)]
impl CodeGenerator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate PTX code for a kernel's instructions.
    ///
    /// # Errors
    ///
    /// Returns `CompileError::UnsupportedOperation` if the instruction stream contains
    /// call instructions or unknown opcodes.
    pub fn generate(
        &mut self,
        instructions: &[DecodedInstruction],
        kernel: &KernelInfo,
    ) -> Result<String, CompileError> {
        self.output.clear();
        self.indent_level = 1;
        self.cf = ControlFlowState::new();

        self.output
            .push_str(&crate::kernel::emit_entry_start(&kernel.name));
        self.output.push_str(&registers::emit_declarations(
            kernel.register_count,
            kernel.local_memory_size,
        ));
        self.line("");

        for instr in instructions {
            self.emit_instruction(instr)?;
        }

        self.output.push_str(crate::kernel::emit_entry_end());

        Ok(self.output.clone())
    }

    fn line(&mut self, text: &str) {
        if text.is_empty() {
            self.output.push('\n');
            return;
        }
        for _ in 0..self.indent_level {
            self.output.push_str("    ");
        }
        writeln!(self.output, "{text}").unwrap();
    }

    fn lines(&mut self, items: &[String]) {
        for item in items {
            if item.ends_with(':') {
                writeln!(self.output, "{item}").unwrap();
            } else {
                self.line(item);
            }
        }
    }

    fn pred_prefix(instr: &DecodedInstruction) -> String {
        if instr.predicate == 0 {
            String::new()
        } else if instr.predicate_negated {
            format!("@!{} ", pred(instr.predicate))
        } else {
            format!("@{} ", pred(instr.predicate))
        }
    }

    fn emit_instruction(&mut self, instr: &DecodedInstruction) -> Result<(), CompileError> {
        let is_control_flow = matches!(
            instr.operation,
            Operation::If { .. }
                | Operation::Else
                | Operation::Endif
                | Operation::Loop
                | Operation::Break { .. }
                | Operation::Continue { .. }
                | Operation::Endloop
        );

        if is_control_flow && instr.predicate != 0 {
            return Err(CompileError::UnsupportedOperation(
                "predicated control flow instructions are not supported in PTX backend"
                    .to_string(),
            ));
        }

        self.emit_operation(instr)?;
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn emit_operation(&mut self, instr: &DecodedInstruction) -> Result<(), CompileError> {
        let pp = Self::pred_prefix(instr);
        match &instr.operation {
            Operation::Iadd { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{pp}add.s32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Isub { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{pp}sub.s32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Imul { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{pp}mul.lo.s32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::ImulHi { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{pp}mul.hi.s32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Imad { rd, rs1, rs2, rs3 } => {
                self.line(&format!(
                    "{pp}mad.lo.s32 {}, {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2),
                    reg(*rs3)
                ));
            }
            Operation::Idiv { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{pp}div.s32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Imod { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{pp}rem.s32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Ineg { rd, rs1 } => {
                self.line(&format!("{pp}neg.s32 {}, {};", reg(*rd), reg(*rs1)));
            }
            Operation::Iabs { rd, rs1 } => {
                self.line(&format!("{pp}abs.s32 {}, {};", reg(*rd), reg(*rs1)));
            }
            Operation::Imin { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{pp}min.s32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Imax { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{pp}max.s32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Iclamp {
                rd,
                rs1,
                rs2,
                rs3,
            } => {
                self.line(&format!(
                    "{pp}max.s32 %t0, {}, {};",
                    reg(*rs1),
                    reg(*rs2)
                ));
                self.line(&format!("{pp}min.s32 {}, %t0, {};", reg(*rd), reg(*rs3)));
            }

            Operation::Fadd { rd, rs1, rs2 } => {
                self.emit_float_binop(&pp, "add.f32", *rd, *rs1, *rs2);
            }
            Operation::Fsub { rd, rs1, rs2 } => {
                self.emit_float_binop(&pp, "sub.f32", *rd, *rs1, *rs2);
            }
            Operation::Fmul { rd, rs1, rs2 } => {
                self.emit_float_binop(&pp, "mul.f32", *rd, *rs1, *rs2);
            }
            Operation::Fma { rd, rs1, rs2, rs3 } => {
                self.line(&format!("{pp}mov.b32 {}, {};", freg(*rs1), reg(*rs1)));
                self.line(&format!("{pp}mov.b32 {}, {};", freg(*rs2), reg(*rs2)));
                self.line(&format!("{pp}mov.b32 {}, {};", freg(*rs3), reg(*rs3)));
                self.line(&format!(
                    "{pp}fma.rn.f32 {}, {}, {}, {};",
                    freg(*rd),
                    freg(*rs1),
                    freg(*rs2),
                    freg(*rs3)
                ));
                self.line(&format!("{pp}mov.b32 {}, {};", reg(*rd), freg(*rd)));
            }
            Operation::Fdiv { rd, rs1, rs2 } => {
                self.emit_float_binop(&pp, "div.approx.f32", *rd, *rs1, *rs2);
            }
            Operation::Fneg { rd, rs1 } => {
                self.emit_float_unaryop(&pp, "neg.f32", *rd, *rs1);
            }
            Operation::Fabs { rd, rs1 } => {
                self.emit_float_unaryop(&pp, "abs.f32", *rd, *rs1);
            }
            Operation::Fmin { rd, rs1, rs2 } => {
                self.emit_float_binop(&pp, "min.f32", *rd, *rs1, *rs2);
            }
            Operation::Fmax { rd, rs1, rs2 } => {
                self.emit_float_binop(&pp, "max.f32", *rd, *rs1, *rs2);
            }
            Operation::Fclamp {
                rd,
                rs1,
                rs2,
                rs3,
            } => {
                self.line(&format!("{pp}mov.b32 {}, {};", freg(*rs1), reg(*rs1)));
                self.line(&format!("{pp}mov.b32 {}, {};", freg(*rs2), reg(*rs2)));
                self.line(&format!("{pp}mov.b32 {}, {};", freg(*rs3), reg(*rs3)));
                self.line(&format!(
                    "{pp}max.f32 %ft0, {}, {};",
                    freg(*rs1),
                    freg(*rs2)
                ));
                self.line(&format!(
                    "{pp}min.f32 {}, %ft0, {};",
                    freg(*rd),
                    freg(*rs3)
                ));
                self.line(&format!("{pp}mov.b32 {}, {};", reg(*rd), freg(*rd)));
            }
            Operation::Fsqrt { rd, rs1 } => {
                self.emit_float_unaryop(&pp, "sqrt.approx.f32", *rd, *rs1);
            }
            Operation::FUnary { op, rd, rs1 } => {
                self.emit_funary(&pp, *op, *rd, *rs1);
            }

            Operation::F16 {
                op,
                rd,
                rs1,
                rs2,
                rs3,
            } => {
                self.emit_f16(&pp, *op, *rd, *rs1, *rs2, *rs3);
            }
            Operation::F16Packed {
                op,
                rd,
                rs1,
                rs2,
                rs3,
            } => {
                self.emit_f16_packed(&pp, *op, *rd, *rs1, *rs2, *rs3);
            }
            Operation::F64 {
                op,
                rd,
                rs1,
                rs2,
                rs3,
            } => {
                self.emit_f64(&pp, *op, *rd, *rs1, *rs2, *rs3);
            }
            Operation::F64DivSqrt { op, rd, rs1, rs2 } => {
                self.emit_f64_div_sqrt(&pp, *op, *rd, *rs1, *rs2);
            }

            Operation::And { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{pp}and.b32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Or { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{pp}or.b32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Xor { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{pp}xor.b32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Not { rd, rs1 } => {
                self.line(&format!("{pp}not.b32 {}, {};", reg(*rd), reg(*rs1)));
            }
            Operation::Shl { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{pp}shl.b32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Shr { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{pp}shr.u32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Sar { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{pp}shr.s32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::BitOp {
                op,
                rd,
                rs1,
                rs2,
                rs3,
                rs4,
            } => {
                self.emit_bitop(&pp, *op, *rd, *rs1, *rs2, *rs3, *rs4);
            }

            Operation::Icmp { op, pd, rs1, rs2 } => {
                self.emit_int_cmp(&pp, "s32", *op, *pd, *rs1, *rs2);
            }
            Operation::Ucmp { op, pd, rs1, rs2 } => {
                self.emit_int_cmp(&pp, "u32", *op, *pd, *rs1, *rs2);
            }
            Operation::Fcmp { op, pd, rs1, rs2 } => {
                self.emit_fcmp(&pp, *op, *pd, *rs1, *rs2);
            }

            Operation::Select { rd, ps, rs1, rs2 } => {
                self.line(&format!(
                    "{pp}selp.b32 {}, {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2),
                    pred(*ps)
                ));
            }
            Operation::Cvt { cvt_type, rd, rs1 } => {
                self.emit_cvt(&pp, *cvt_type, *rd, *rs1);
            }

            Operation::LocalLoad { width, rd, addr } => {
                let stmts = memory::emit_shared_load(*width, *rd, *addr);
                self.lines(&stmts);
            }
            Operation::LocalStore {
                width,
                addr,
                value,
            } => {
                let stmts = memory::emit_shared_store(*width, *addr, *value);
                self.lines(&stmts);
            }
            Operation::DeviceLoad { width, rd, addr } => {
                let stmts = memory::emit_global_load(*width, *rd, *addr);
                self.lines(&stmts);
            }
            Operation::DeviceStore {
                width,
                addr,
                value,
            } => {
                let stmts = memory::emit_global_store(*width, *addr, *value);
                self.lines(&stmts);
            }

            Operation::LocalAtomic {
                op,
                rd,
                addr,
                value,
            } => {
                let stmts = memory::emit_shared_atomic(*op, *rd, *addr, *value);
                self.lines(&stmts);
            }
            Operation::LocalAtomicCas {
                rd,
                addr,
                expected,
                desired,
            } => {
                let stmts = memory::emit_shared_atomic_cas(*rd, *addr, *expected, *desired);
                self.lines(&stmts);
            }
            Operation::DeviceAtomic {
                op,
                rd,
                addr,
                value,
                ..
            } => {
                let stmts = memory::emit_global_atomic(*op, *rd, *addr, *value);
                self.lines(&stmts);
            }
            Operation::DeviceAtomicCas {
                rd,
                addr,
                expected,
                desired,
                ..
            } => {
                let stmts = memory::emit_global_atomic_cas(*rd, *addr, *expected, *desired);
                self.lines(&stmts);
            }

            Operation::WaveOp { op, rd, rs1, rs2 } => {
                let stmts = intrinsics::emit_wave_op(*op, *rd, *rs1, *rs2);
                self.lines(&stmts);
            }
            Operation::WaveReduce { op, rd, rs1 } => {
                let stmts = intrinsics::emit_wave_reduce(*op, *rd, *rs1);
                self.lines(&stmts);
            }
            Operation::WaveBallot { rd, ps } => {
                let stmts = intrinsics::emit_wave_ballot(*rd, *ps);
                self.lines(&stmts);
            }
            Operation::WaveVote { op, pd, ps } => {
                let stmts = intrinsics::emit_wave_vote(*op, *pd, *ps);
                self.lines(&stmts);
            }

            Operation::If { ps } => {
                let stmts = self.cf.emit_if(*ps);
                self.lines(&stmts);
            }
            Operation::Else => {
                let stmts = self.cf.emit_else();
                self.lines(&stmts);
            }
            Operation::Endif => {
                let stmts = self.cf.emit_endif();
                self.lines(&stmts);
            }
            Operation::Loop => {
                let stmts = self.cf.emit_loop();
                self.lines(&stmts);
            }
            Operation::Break { ps } => {
                let stmts = self.cf.emit_break(*ps);
                self.lines(&stmts);
            }
            Operation::Continue { ps } => {
                let stmts = self.cf.emit_continue(*ps);
                self.lines(&stmts);
            }
            Operation::Endloop => {
                let stmts = self.cf.emit_endloop();
                self.lines(&stmts);
            }
            Operation::Call { .. } => {
                return Err(CompileError::UnsupportedOperation(
                    "call instructions are not supported in the PTX backend".to_string(),
                ));
            }

            Operation::Return | Operation::Halt => {
                self.line(&format!("{pp}ret;"));
            }
            Operation::Barrier => {
                self.line(&format!("{pp}bar.sync 0;"));
            }
            Operation::FenceAcquire { scope } | Operation::FenceRelease { scope } | Operation::FenceAcqRel { scope } => {
                let fence = match scope {
                    wave_decode::Scope::Wave | wave_decode::Scope::Workgroup => "membar.cta;",
                    wave_decode::Scope::Device => "membar.gl;",
                    wave_decode::Scope::System => "membar.sys;",
                };
                self.line(&format!("{pp}{fence}"));
            }
            Operation::Wait | Operation::Nop => {}

            Operation::Mov { rd, rs1 } => {
                self.line(&format!("{pp}mov.b32 {}, {};", reg(*rd), reg(*rs1)));
            }
            Operation::MovImm { rd, imm } => {
                self.line(&format!("{pp}mov.b32 {}, {};", reg(*rd), imm));
            }
            Operation::MovSr { rd, sr_index } => {
                let stmts = registers::emit_special_reg(*rd, *sr_index);
                for s in &stmts {
                    self.line(&format!("{pp}{s}"));
                }
            }

            Operation::Unknown { opcode, .. } => {
                return Err(CompileError::UnsupportedOperation(format!(
                    "unknown opcode 0x{opcode:02x}"
                )));
            }
        }
        Ok(())
    }

    fn emit_float_binop(&mut self, pp: &str, op: &str, rd: u8, rs1: u8, rs2: u8) {
        self.line(&format!("{pp}mov.b32 {}, {};", freg(rs1), reg(rs1)));
        self.line(&format!("{pp}mov.b32 {}, {};", freg(rs2), reg(rs2)));
        self.line(&format!(
            "{pp}{op} {}, {}, {};",
            freg(rd),
            freg(rs1),
            freg(rs2)
        ));
        self.line(&format!("{pp}mov.b32 {}, {};", reg(rd), freg(rd)));
    }

    fn emit_float_unaryop(&mut self, pp: &str, op: &str, rd: u8, rs1: u8) {
        self.line(&format!("{pp}mov.b32 {}, {};", freg(rs1), reg(rs1)));
        self.line(&format!("{pp}{op} {}, {};", freg(rd), freg(rs1)));
        self.line(&format!("{pp}mov.b32 {}, {};", reg(rd), freg(rd)));
    }

    fn emit_funary(&mut self, pp: &str, op: FUnaryOp, rd: u8, rs1: u8) {
        let ptx_op = match op {
            FUnaryOp::Frsqrt => "rsqrt.approx.f32",
            FUnaryOp::Frcp => "rcp.approx.f32",
            FUnaryOp::Ffloor => "cvt.rmi.f32.f32",
            FUnaryOp::Fceil => "cvt.rpi.f32.f32",
            FUnaryOp::Fround => "cvt.rni.f32.f32",
            FUnaryOp::Ftrunc => "cvt.rzi.f32.f32",
            FUnaryOp::Fsin => "sin.approx.f32",
            FUnaryOp::Fcos => "cos.approx.f32",
            FUnaryOp::Fexp2 => "ex2.approx.f32",
            FUnaryOp::Flog2 => "lg2.approx.f32",
            FUnaryOp::Ffract | FUnaryOp::Fsat => {
                self.emit_funary_multi(pp, op, rd, rs1);
                return;
            }
        };
        self.emit_float_unaryop(pp, ptx_op, rd, rs1);
    }

    fn emit_funary_multi(&mut self, pp: &str, op: FUnaryOp, rd: u8, rs1: u8) {
        self.line(&format!("{pp}mov.b32 {}, {};", freg(rs1), reg(rs1)));
        match op {
            FUnaryOp::Ffract => {
                self.line(&format!(
                    "{pp}cvt.rmi.f32.f32 %ft0, {};",
                    freg(rs1)
                ));
                self.line(&format!(
                    "{pp}sub.f32 {}, {}, %ft0;",
                    freg(rd),
                    freg(rs1)
                ));
            }
            FUnaryOp::Fsat => {
                self.line(&format!(
                    "{pp}max.f32 %ft0, {}, 0f00000000;",
                    freg(rs1)
                ));
                self.line(&format!("{pp}min.f32 {}, %ft0, 0f3F800000;", freg(rd)));
            }
            _ => {}
        }
        self.line(&format!("{pp}mov.b32 {}, {};", reg(rd), freg(rd)));
    }

    fn emit_f16(&mut self, pp: &str, op: F16Op, rd: u8, rs1: u8, rs2: u8, rs3: Option<u8>) {
        self.line(&format!("{pp}cvt.u16.u32 %t0, {};", reg(rs1)));
        self.line(&format!("{pp}cvt.u16.u32 %t1, {};", reg(rs2)));

        let ptx_op = match op {
            F16Op::Hadd => "add.f16",
            F16Op::Hsub => "sub.f16",
            F16Op::Hmul => "mul.f16",
            F16Op::Hma => {
                self.line(&format!(
                    "{pp}cvt.u16.u32 %t2, {};",
                    reg(rs3.unwrap_or(0))
                ));
                self.line(&format!("{pp}fma.rn.f16 %t0, %t0, %t1, %t2;"));
                self.line(&format!("{pp}cvt.u32.u16 {}, %t0;", reg(rd)));
                return;
            }
        };

        self.line(&format!("{pp}{ptx_op} %t0, %t0, %t1;"));
        self.line(&format!("{pp}cvt.u32.u16 {}, %t0;", reg(rd)));
    }

    fn emit_f16_packed(
        &mut self,
        pp: &str,
        op: F16PackedOp,
        rd: u8,
        rs1: u8,
        rs2: u8,
        rs3: Option<u8>,
    ) {
        let ptx_op = match op {
            F16PackedOp::Hadd2 => "add.f16x2",
            F16PackedOp::Hmul2 => "mul.f16x2",
            F16PackedOp::Hma2 => {
                self.line(&format!(
                    "{pp}fma.rn.f16x2 {}, {}, {}, {};",
                    reg(rd),
                    reg(rs1),
                    reg(rs2),
                    reg(rs3.unwrap_or(0))
                ));
                return;
            }
        };

        self.line(&format!(
            "{pp}{ptx_op} {}, {}, {};",
            reg(rd),
            reg(rs1),
            reg(rs2)
        ));
    }

    fn emit_f64(&mut self, pp: &str, op: F64Op, rd: u8, rs1: u8, rs2: u8, rs3: Option<u8>) {
        self.line(&format!(
            "{pp}mov.b64 %rd2, {{{}, {}}};",
            reg(rs1),
            reg(rs1 + 1)
        ));
        self.line(&format!(
            "{pp}mov.b64 %rd3, {{{}, {}}};",
            reg(rs2),
            reg(rs2 + 1)
        ));

        let ptx_op = match op {
            F64Op::Dadd => "add.f64",
            F64Op::Dsub => "sub.f64",
            F64Op::Dmul => "mul.f64",
            F64Op::Dma => {
                let s3 = rs3.unwrap_or(0);
                self.line(&format!(
                    "{pp}mov.b64 %rd1, {{{}, {}}};",
                    reg(s3),
                    reg(s3 + 1)
                ));
                self.line(&format!("{pp}fma.rn.f64 %rd2, %rd2, %rd3, %rd1;"));
                self.line(&format!(
                    "{pp}mov.b64 {{{}, {}}}, %rd2;",
                    reg(rd),
                    reg(rd + 1)
                ));
                return;
            }
        };

        self.line(&format!("{pp}{ptx_op} %rd2, %rd2, %rd3;"));
        self.line(&format!(
            "{pp}mov.b64 {{{}, {}}}, %rd2;",
            reg(rd),
            reg(rd + 1)
        ));
    }

    fn emit_f64_div_sqrt(
        &mut self,
        pp: &str,
        op: F64DivSqrtOp,
        rd: u8,
        rs1: u8,
        rs2: Option<u8>,
    ) {
        self.line(&format!(
            "{pp}mov.b64 %rd2, {{{}, {}}};",
            reg(rs1),
            reg(rs1 + 1)
        ));

        match op {
            F64DivSqrtOp::Ddiv => {
                let s2 = rs2.unwrap_or(0);
                self.line(&format!(
                    "{pp}mov.b64 %rd3, {{{}, {}}};",
                    reg(s2),
                    reg(s2 + 1)
                ));
                self.line(&format!("{pp}div.rn.f64 %rd2, %rd2, %rd3;"));
            }
            F64DivSqrtOp::Dsqrt => {
                self.line(&format!("{pp}sqrt.rn.f64 %rd2, %rd2;"));
            }
        }

        self.line(&format!(
            "{pp}mov.b64 {{{}, {}}}, %rd2;",
            reg(rd),
            reg(rd + 1)
        ));
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_bitop(
        &mut self,
        pp: &str,
        op: BitOpType,
        rd: u8,
        rs1: u8,
        rs2: Option<u8>,
        rs3: Option<u8>,
        rs4: Option<u8>,
    ) {
        match op {
            BitOpType::Bitcount => {
                self.line(&format!("{pp}popc.b32 {}, {};", reg(rd), reg(rs1)));
            }
            BitOpType::Bitfind => {
                self.line(&format!("{pp}neg.s32 %t0, {};", reg(rs1)));
                self.line(&format!("{pp}and.b32 %t0, {}, %t0;", reg(rs1)));
                self.line(&format!("{pp}bfind.u32 {}, %t0;", reg(rd)));
            }
            BitOpType::Bitrev => {
                self.line(&format!("{pp}brev.b32 {}, {};", reg(rd), reg(rs1)));
            }
            BitOpType::Bfe => {
                self.line(&format!(
                    "{pp}bfe.u32 {}, {}, {}, {};",
                    reg(rd),
                    reg(rs1),
                    reg(rs2.unwrap_or(0)),
                    reg(rs3.unwrap_or(0))
                ));
            }
            BitOpType::Bfi => {
                self.line(&format!(
                    "{pp}bfi.b32 {}, {}, {}, {}, {};",
                    reg(rd),
                    reg(rs1),
                    reg(rs2.unwrap_or(0)),
                    reg(rs3.unwrap_or(0)),
                    reg(rs4.unwrap_or(0))
                ));
            }
        }
    }

    fn emit_int_cmp(&mut self, pp: &str, ty: &str, op: CmpOp, pd: u8, rs1: u8, rs2: u8) {
        let cmp = match op {
            CmpOp::Eq => "eq",
            CmpOp::Ne => "ne",
            CmpOp::Lt => "lt",
            CmpOp::Le => "le",
            CmpOp::Gt => "gt",
            CmpOp::Ge => "ge",
            CmpOp::Ord | CmpOp::Unord => {
                self.emit_fcmp(pp, op, pd, rs1, rs2);
                return;
            }
        };

        self.line(&format!(
            "{pp}setp.{cmp}.{ty} {}, {}, {};",
            pred(pd),
            reg(rs1),
            reg(rs2)
        ));
    }

    fn emit_fcmp(&mut self, pp: &str, op: CmpOp, pd: u8, rs1: u8, rs2: u8) {
        self.line(&format!("{pp}mov.b32 {}, {};", freg(rs1), reg(rs1)));
        self.line(&format!("{pp}mov.b32 {}, {};", freg(rs2), reg(rs2)));

        let cmp = match op {
            CmpOp::Eq => "eq",
            CmpOp::Ne => "ne",
            CmpOp::Lt => "lt",
            CmpOp::Le => "le",
            CmpOp::Gt => "gt",
            CmpOp::Ge => "ge",
            CmpOp::Ord => "num",
            CmpOp::Unord => "nan",
        };

        self.line(&format!(
            "{pp}setp.{cmp}.f32 {}, {}, {};",
            pred(pd),
            freg(rs1),
            freg(rs2)
        ));
    }

    fn emit_cvt(&mut self, pp: &str, cvt_type: CvtType, rd: u8, rs1: u8) {
        match cvt_type {
            CvtType::F32I32 => {
                self.line(&format!("{pp}mov.b32 {}, {};", freg(rs1), reg(rs1)));
                self.line(&format!(
                    "{pp}cvt.rzi.s32.f32 {}, {};",
                    reg(rd),
                    freg(rs1)
                ));
            }
            CvtType::F32U32 => {
                self.line(&format!("{pp}mov.b32 {}, {};", freg(rs1), reg(rs1)));
                self.line(&format!(
                    "{pp}cvt.rzi.u32.f32 {}, {};",
                    reg(rd),
                    freg(rs1)
                ));
            }
            CvtType::I32F32 => {
                self.line(&format!(
                    "{pp}cvt.rn.f32.s32 {}, {};",
                    freg(rd),
                    reg(rs1)
                ));
                self.line(&format!("{pp}mov.b32 {}, {};", reg(rd), freg(rd)));
            }
            CvtType::U32F32 => {
                self.line(&format!(
                    "{pp}cvt.rn.f32.u32 {}, {};",
                    freg(rd),
                    reg(rs1)
                ));
                self.line(&format!("{pp}mov.b32 {}, {};", reg(rd), freg(rd)));
            }
            CvtType::F32F16 => {
                self.line(&format!("{pp}mov.b32 {}, {};", freg(rs1), reg(rs1)));
                self.line(&format!(
                    "{pp}cvt.rn.f16.f32 %t0, {};",
                    freg(rs1)
                ));
                self.line(&format!("{pp}cvt.u32.u16 {}, %t0;", reg(rd)));
            }
            CvtType::F16F32 => {
                self.line(&format!("{pp}cvt.u16.u32 %t0, {};", reg(rs1)));
                self.line(&format!("{pp}cvt.f32.f16 {}, %t0;", freg(rd)));
                self.line(&format!("{pp}mov.b32 {}, {};", reg(rd), freg(rd)));
            }
            CvtType::F32F64 => {
                self.line(&format!("{pp}mov.b32 {}, {};", freg(rs1), reg(rs1)));
                self.line(&format!(
                    "{pp}cvt.f64.f32 %rd2, {};",
                    freg(rs1)
                ));
                self.line(&format!(
                    "{pp}mov.b64 {{{}, {}}}, %rd2;",
                    reg(rd),
                    reg(rd + 1)
                ));
            }
            CvtType::F64F32 => {
                self.line(&format!(
                    "{pp}mov.b64 %rd2, {{{}, {}}};",
                    reg(rs1),
                    reg(rs1 + 1)
                ));
                self.line(&format!("{pp}cvt.rn.f32.f64 {}, %rd2;", freg(rd)));
                self.line(&format!("{pp}mov.b32 {}, {};", reg(rd), freg(rd)));
            }
        }
    }
}
