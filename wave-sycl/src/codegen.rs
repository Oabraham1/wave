// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Translate decoded WAVE operations into SYCL C++.

use std::fmt::Write;

use wave_decode::opcodes::{
    Bf16Op, Bf16PackedOp, BitOpType, CmpOp, CvtType, F16Op, F16PackedOp, F64DivSqrtOp, F64Op,
    FUnaryOp,
};
use wave_decode::{DecodedInstruction, KernelInfo, Operation};

use crate::control_flow;
use crate::intrinsics;
use crate::memory;
use crate::registers::{self, pred, reg};
use crate::CompileError;

#[derive(Debug)]
pub struct CodeGenerator {
    output: String,
    indent_level: usize,
}

impl Default for CodeGenerator {
    fn default() -> Self {
        Self {
            output: String::new(),
            indent_level: 4,
        }
    }
}

impl CodeGenerator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate SYCL code for a kernel's instructions.
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
        self.indent_level = 4;

        self.output.push_str(&crate::kernel::emit_launch_start(
            &kernel.name,
            kernel.local_memory_size,
        ));

        self.output.push_str(&registers::emit_declarations(
            kernel.register_count,
            kernel.local_memory_size > 0,
        ));
        self.line("");

        for instr in instructions {
            self.emit_instruction(instr)?;
        }

        self.output.push_str(crate::kernel::emit_launch_end());

        Ok(std::mem::take(&mut self.output))
    }

    fn line(&mut self, text: &str) {
        if text.is_empty() {
            self.output.push('\n');
            return;
        }
        self.line_args(format_args!("{text}"));
    }

    fn line_args(&mut self, args: std::fmt::Arguments<'_>) {
        for _ in 0..self.indent_level {
            self.output.push_str("    ");
        }
        writeln!(self.output, "{args}").expect("writing to String cannot fail");
    }

    fn lines(&mut self, items: &[String]) {
        for item in items {
            self.line(item);
        }
    }

    fn emit_instruction(&mut self, instr: &DecodedInstruction) -> Result<(), CompileError> {
        if instr.predicate != 0 {
            let guard = if instr.predicate_negated {
                format!("if (!{}) {{", pred(instr.predicate))
            } else {
                format!("if ({}) {{", pred(instr.predicate))
            };
            self.line(&guard);
            self.indent_level += 1;
            self.emit_operation(&instr.operation)?;
            self.indent_level = self.indent_level.saturating_sub(1);
            self.line("}");
        } else {
            self.emit_operation(&instr.operation)?;
        }
        Ok(())
    }

    fn emit_operation(&mut self, op: &Operation) -> Result<(), CompileError> {
        match op {
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
            | Operation::Iclamp { .. } => self.emit_operation_integer(op),
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
            | Operation::FUnary { .. } => self.emit_operation_float(op),
            Operation::F16 { .. }
            | Operation::F16Packed { .. }
            | Operation::Bf16 { .. }
            | Operation::Bf16Packed { .. } => self.emit_operation_half(op),
            Operation::F64 { .. } | Operation::F64DivSqrt { .. } => {
                self.emit_operation_double(op);
            }
            Operation::And { .. }
            | Operation::Or { .. }
            | Operation::Xor { .. }
            | Operation::Not { .. }
            | Operation::Shl { .. }
            | Operation::Shr { .. }
            | Operation::Sar { .. }
            | Operation::BitOp { .. } => self.emit_operation_bitwise(op),
            Operation::Icmp { .. }
            | Operation::Ucmp { .. }
            | Operation::Fcmp { .. }
            | Operation::Select { .. }
            | Operation::Cvt { .. } => self.emit_operation_compare(op),
            Operation::LocalLoad { .. }
            | Operation::LocalStore { .. }
            | Operation::DeviceLoad { .. }
            | Operation::DeviceStore { .. } => self.emit_operation_load_store(op),
            Operation::LocalAtomic { .. }
            | Operation::LocalAtomicCas { .. }
            | Operation::DeviceAtomic { .. }
            | Operation::DeviceAtomicCas { .. } => self.emit_operation_atomic(op),
            Operation::WaveOp { .. }
            | Operation::WaveReduce { .. }
            | Operation::WaveBallot { .. }
            | Operation::WaveVote { .. }
            | Operation::MmaLoadA { .. }
            | Operation::MmaLoadB { .. }
            | Operation::MmaStoreC { .. }
            | Operation::MmaCompute { .. } => self.emit_operation_wave(op),
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
            | Operation::Nop => self.emit_operation_control(op)?,
            Operation::Mov { .. }
            | Operation::MovImm { .. }
            | Operation::MovSr { .. }
            | Operation::Unknown { .. } => self.emit_operation_move(op)?,
        }
        Ok(())
    }

    fn emit_operation_integer(&mut self, op: &Operation) {
        match op {
            Operation::Iadd { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = {} + {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Isub { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = {} - {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Imul { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = {} * {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::ImulHi { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = (uint32_t)(((uint64_t){} * (uint64_t){}) >> 32);",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Imad { rd, rs1, rs2, rs3 } => {
                self.line_args(format_args!(
                    "{} = {} * {} + {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2),
                    reg(*rs3)
                ));
            }
            Operation::Idiv { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = (uint32_t)((int32_t){} / (int32_t){});",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Imod { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = (uint32_t)((int32_t){} % (int32_t){});",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Ineg { rd, rs1 } => {
                self.line_args(format_args!(
                    "{} = (uint32_t)(-(int32_t){});",
                    reg(*rd),
                    reg(*rs1)
                ));
            }
            Operation::Iabs { rd, rs1 } => {
                self.line_args(format_args!(
                    "{} = (uint32_t)sycl::abs((int32_t){});",
                    reg(*rd),
                    reg(*rs1)
                ));
            }
            Operation::Imin { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = (uint32_t)sycl::min((int32_t){}, (int32_t){});",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Imax { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = (uint32_t)sycl::max((int32_t){}, (int32_t){});",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Iclamp { rd, rs1, rs2, rs3 } => {
                self.line_args(format_args!(
                    "{} = (uint32_t)sycl::clamp((int32_t){}, (int32_t){}, (int32_t){});",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2),
                    reg(*rs3)
                ));
            }
            _ => unreachable!("emit_operation routes only integer operations here"),
        }
    }

    fn emit_operation_float(&mut self, op: &Operation) {
        match op {
            Operation::Fadd { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = ri(rf({}) + rf({}));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Fsub { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = ri(rf({}) - rf({}));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Fmul { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = ri(rf({}) * rf({}));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Fma { rd, rs1, rs2, rs3 } => {
                self.line_args(format_args!(
                    "{} = ri(sycl::fma(rf({}), rf({}), rf({})));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2),
                    reg(*rs3)
                ));
            }
            Operation::Fdiv { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = ri(rf({}) / rf({}));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Fneg { rd, rs1 } => {
                self.line_args(format_args!("{} = ri(-rf({}));", reg(*rd), reg(*rs1)));
            }
            Operation::Fabs { rd, rs1 } => {
                self.line_args(format_args!(
                    "{} = ri(sycl::fabs(rf({})));",
                    reg(*rd),
                    reg(*rs1)
                ));
            }
            Operation::Fmin { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = ri(sycl::fmin(rf({}), rf({})));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Fmax { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = ri(sycl::fmax(rf({}), rf({})));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Fclamp { rd, rs1, rs2, rs3 } => {
                self.line_args(format_args!(
                    "{} = ri(sycl::clamp(rf({}), rf({}), rf({})));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2),
                    reg(*rs3)
                ));
            }
            Operation::Fsqrt { rd, rs1 } => {
                self.line_args(format_args!(
                    "{} = ri(sycl::sqrt(rf({})));",
                    reg(*rd),
                    reg(*rs1)
                ));
            }
            Operation::FUnary { op, rd, rs1 } => {
                self.emit_funary(*op, *rd, *rs1);
            }
            _ => unreachable!("emit_operation routes only float operations here"),
        }
    }

    fn emit_operation_half(&mut self, op: &Operation) {
        match op {
            Operation::F16 {
                op,
                rd,
                rs1,
                rs2,
                rs3,
            } => {
                self.emit_f16(*op, *rd, *rs1, *rs2, *rs3);
            }
            Operation::F16Packed {
                op, rd, rs1, rs2, ..
            } => {
                self.emit_f16_packed(*op, *rd, *rs1, *rs2);
            }
            Operation::Bf16 {
                op,
                rd,
                rs1,
                rs2,
                rs3,
            } => {
                self.emit_bf16(*op, *rd, *rs1, *rs2, *rs3);
            }
            Operation::Bf16Packed {
                op, rd, rs1, rs2, ..
            } => {
                self.emit_bf16_packed(*op, *rd, *rs1, *rs2);
            }
            _ => unreachable!("emit_operation routes only half operations here"),
        }
    }

    fn emit_operation_double(&mut self, op: &Operation) {
        match op {
            Operation::F64 {
                op,
                rd,
                rs1,
                rs2,
                rs3,
            } => {
                self.emit_f64(*op, *rd, *rs1, *rs2, *rs3);
            }
            Operation::F64DivSqrt { op, rd, rs1, rs2 } => {
                self.emit_f64_div_sqrt(*op, *rd, *rs1, *rs2);
            }
            _ => unreachable!("emit_operation routes only double operations here"),
        }
    }

    fn emit_operation_bitwise(&mut self, op: &Operation) {
        match op {
            Operation::And { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = {} & {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Or { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = {} | {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Xor { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = {} ^ {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Not { rd, rs1 } => {
                self.line_args(format_args!("{} = ~{};", reg(*rd), reg(*rs1)));
            }
            Operation::Shl { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = {} << ({} & 0x1Fu);",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Shr { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = {} >> ({} & 0x1Fu);",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Sar { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = (uint32_t)((int32_t){} >> ({} & 0x1Fu));",
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
                self.emit_bitop(*op, *rd, *rs1, *rs2, *rs3, *rs4);
            }
            _ => unreachable!("emit_operation routes only bitwise operations here"),
        }
    }

    fn emit_operation_compare(&mut self, op: &Operation) {
        match op {
            Operation::Icmp { op, pd, rs1, rs2 } => {
                self.emit_cmp("(int32_t)", *op, *pd, *rs1, *rs2);
            }
            Operation::Ucmp { op, pd, rs1, rs2 } => {
                self.emit_cmp("", *op, *pd, *rs1, *rs2);
            }
            Operation::Fcmp { op, pd, rs1, rs2 } => {
                self.emit_fcmp(*op, *pd, *rs1, *rs2);
            }
            Operation::Select { rd, ps, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = {} ? {} : {};",
                    reg(*rd),
                    pred(*ps),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Cvt { cvt_type, rd, rs1 } => {
                self.emit_cvt(*cvt_type, *rd, *rs1);
            }
            _ => unreachable!("emit_operation routes only compare operations here"),
        }
    }

    fn emit_operation_load_store(&mut self, op: &Operation) {
        match op {
            Operation::LocalLoad { width, rd, addr } => {
                let stmts = memory::emit_load("lm", *width, *rd, *addr);
                self.lines(&stmts);
            }
            Operation::LocalStore { width, addr, value } => {
                let stmts = memory::emit_store("lm", *width, *addr, *value);
                self.lines(&stmts);
            }
            Operation::DeviceLoad { width, rd, addr } => {
                let stmts = memory::emit_load("device_mem_usm", *width, *rd, *addr);
                self.lines(&stmts);
            }
            Operation::DeviceStore { width, addr, value } => {
                let stmts = memory::emit_store("device_mem_usm", *width, *addr, *value);
                self.lines(&stmts);
            }
            _ => unreachable!("emit_operation routes only load_store operations here"),
        }
    }

    fn emit_operation_atomic(&mut self, op: &Operation) {
        match op {
            Operation::LocalAtomic {
                op,
                rd,
                addr,
                value,
            } => {
                let stmts = memory::emit_atomic("lm", *op, *rd, *addr, *value);
                self.lines(&stmts);
            }
            Operation::LocalAtomicCas {
                rd,
                addr,
                expected,
                desired,
            } => {
                let stmts = memory::emit_atomic_cas("lm", *rd, *addr, *expected, *desired);
                self.lines(&stmts);
            }
            Operation::DeviceAtomic {
                op,
                rd,
                addr,
                value,
                ..
            } => {
                let stmts = memory::emit_atomic("device_mem_usm", *op, *rd, *addr, *value);
                self.lines(&stmts);
            }
            Operation::DeviceAtomicCas {
                rd,
                addr,
                expected,
                desired,
                ..
            } => {
                let stmts =
                    memory::emit_atomic_cas("device_mem_usm", *rd, *addr, *expected, *desired);
                self.lines(&stmts);
            }
            _ => unreachable!("emit_operation routes only atomic operations here"),
        }
    }

    fn emit_operation_wave(&mut self, op: &Operation) {
        match op {
            Operation::WaveOp { op, rd, rs1, rs2 } => {
                let stmt = intrinsics::emit_wave_op(*op, *rd, *rs1, *rs2);
                if !stmt.is_empty() {
                    self.line(&stmt);
                }
            }
            Operation::WaveReduce { op, rd, rs1 } => {
                self.line(&intrinsics::emit_wave_reduce(*op, *rd, *rs1));
            }
            Operation::WaveBallot { rd, ps } => {
                self.line(&intrinsics::emit_wave_ballot(*rd, *ps));
            }
            Operation::WaveVote { op, pd, ps } => {
                let stmt = intrinsics::emit_wave_vote(*op, *pd, *ps);
                if !stmt.is_empty() {
                    self.line(&stmt);
                }
            }
            Operation::MmaLoadA { rd: _, rs1, rs2 } => {
                self.line("{");
                self.line_args(format_args!("    uint32_t _addr = {};", reg(*rs1)));
                self.line_args(format_args!("    uint32_t _stride = {};", reg(*rs2)));
                self.line("    for (int _r = 0; _r < 4; _r++) {");
                self.line("        for (int _c = 0; _c < 4; _c++) {");
                self.line("            _mma_a[_r * 4 + _c] = rf(*(uint32_t*)(lm + _addr + _r * _stride + _c * 4));");
                self.line("        }");
                self.line("    }");
                self.line("}");
            }
            Operation::MmaLoadB { rd: _, rs1, rs2 } => {
                self.line("{");
                self.line_args(format_args!("    uint32_t _addr = {};", reg(*rs1)));
                self.line_args(format_args!("    uint32_t _stride = {};", reg(*rs2)));
                self.line("    for (int _r = 0; _r < 4; _r++) {");
                self.line("        for (int _c = 0; _c < 4; _c++) {");
                self.line("            _mma_b[_r * 4 + _c] = rf(*(uint32_t*)(lm + _addr + _r * _stride + _c * 4));");
                self.line("        }");
                self.line("    }");
                self.line("}");
            }
            Operation::MmaStoreC { rd, rs1, rs2: _ } => {
                self.line("{");
                self.line_args(format_args!("    uint32_t _addr = {};", reg(*rd)));
                self.line_args(format_args!("    uint32_t _stride = {};", reg(*rs1)));
                self.line("    for (int _r = 0; _r < 4; _r++) {");
                self.line("        for (int _c = 0; _c < 4; _c++) {");
                self.line("            *(uint32_t*)(lm + _addr + _r * _stride + _c * 4) = ri(_mma_c[_r * 4 + _c]);");
                self.line("        }");
                self.line("    }");
                self.line("}");
            }
            Operation::MmaCompute { .. } => {
                self.line("for (int _i = 0; _i < 4; _i++) {");
                self.line("    for (int _j = 0; _j < 4; _j++) {");
                self.line("        for (int _k = 0; _k < 4; _k++) {");
                self.line("            _mma_c[_i * 4 + _j] = sycl::fma(_mma_a[_i * 4 + _k], _mma_b[_k * 4 + _j], _mma_c[_i * 4 + _j]);");
                self.line("        }");
                self.line("    }");
                self.line("}");
            }
            _ => unreachable!("emit_operation routes only wave operations here"),
        }
    }

    fn emit_operation_control(&mut self, op: &Operation) -> Result<(), CompileError> {
        match op {
            Operation::If { ps } => {
                self.line(&control_flow::emit_if(*ps));
                self.indent_level += 1;
            }
            Operation::Else => {
                self.indent_level = self.indent_level.saturating_sub(1);
                self.line(control_flow::emit_else());
                self.indent_level += 1;
            }
            Operation::Endif => {
                self.indent_level = self.indent_level.saturating_sub(1);
                self.line(control_flow::emit_endif());
            }
            Operation::Loop => {
                self.line(control_flow::emit_loop());
                self.indent_level += 1;
            }
            Operation::Break { ps } => {
                self.line(&control_flow::emit_break(*ps));
            }
            Operation::Continue { ps } => {
                self.line(&control_flow::emit_continue(*ps));
            }
            Operation::Endloop => {
                self.indent_level = self.indent_level.saturating_sub(1);
                self.line(control_flow::emit_endloop());
            }
            Operation::Call { .. } => {
                return Err(CompileError::UnsupportedOperation(
                    "call instructions are not supported in the SYCL backend".to_string(),
                ));
            }
            Operation::Return | Operation::Halt => {
                self.line("return;");
            }
            Operation::Barrier => {
                self.line("group_barrier(it.get_group());");
            }
            Operation::FenceAcquire { scope } => {
                self.emit_fence("memory_order::acquire", *scope);
            }
            Operation::FenceRelease { scope } => {
                self.emit_fence("memory_order::release", *scope);
            }
            Operation::FenceAcqRel { scope } => {
                self.emit_fence("memory_order::acq_rel", *scope);
            }
            Operation::Wait | Operation::Nop => {}
            _ => unreachable!("emit_operation routes only control operations here"),
        }
        Ok(())
    }

    fn emit_operation_move(&mut self, op: &Operation) -> Result<(), CompileError> {
        match op {
            Operation::Mov { rd, rs1 } => {
                self.line_args(format_args!("{} = {};", reg(*rd), reg(*rs1)));
            }
            Operation::MovImm { rd, imm } => {
                self.line_args(format_args!("{} = {imm}u;", reg(*rd)));
            }
            Operation::MovSr { rd, sr_index } => {
                self.line_args(format_args!(
                    "{} = (uint32_t){};",
                    reg(*rd),
                    registers::special_reg_expr(*sr_index)
                ));
            }
            Operation::Unknown { opcode, .. } => {
                return Err(CompileError::UnsupportedOperation(format!(
                    "unknown opcode 0x{opcode:02x}"
                )));
            }
            _ => unreachable!("emit_operation routes only move operations here"),
        }
        Ok(())
    }

    fn emit_fence(&mut self, order: &str, scope: wave_decode::Scope) {
        let sycl_scope = match scope {
            wave_decode::Scope::Wave => "memory_scope::sub_group",
            wave_decode::Scope::Workgroup => "memory_scope::work_group",
            wave_decode::Scope::Device => "memory_scope::device",
            wave_decode::Scope::System => "memory_scope::system",
        };
        self.line_args(format_args!("atomic_fence({order}, {sycl_scope});"));
    }

    fn emit_funary(&mut self, op: FUnaryOp, rd: u8, rs1: u8) {
        let r_d = reg(rd);
        let source1 = reg(rs1);
        let expr = match op {
            FUnaryOp::Frsqrt => format!("{r_d} = ri(sycl::rsqrt(rf({source1})));"),
            FUnaryOp::Frcp => format!("{r_d} = ri(1.0f / rf({source1}));"),
            FUnaryOp::Ffloor => format!("{r_d} = ri(sycl::floor(rf({source1})));"),
            FUnaryOp::Fceil => format!("{r_d} = ri(sycl::ceil(rf({source1})));"),
            FUnaryOp::Fround => format!("{r_d} = ri(sycl::rint(rf({source1})));"),
            FUnaryOp::Ftrunc => format!("{r_d} = ri(sycl::trunc(rf({source1})));"),
            FUnaryOp::Ffract => format!("{r_d} = ri(rf({source1}) - sycl::floor(rf({source1})));"),
            FUnaryOp::Fsat => format!("{r_d} = ri(sycl::clamp(rf({source1}), 0.0f, 1.0f));"),
            FUnaryOp::Fsin => format!("{r_d} = ri(sycl::sin(rf({source1})));"),
            FUnaryOp::Fcos => format!("{r_d} = ri(sycl::cos(rf({source1})));"),
            FUnaryOp::Fexp2 => format!("{r_d} = ri(sycl::exp2(rf({source1})));"),
            FUnaryOp::Flog2 => format!("{r_d} = ri(sycl::log2(rf({source1})));"),
        };
        self.line(&expr);
    }

    fn emit_f16(&mut self, op: F16Op, rd: u8, rs1: u8, rs2: u8, rs3: Option<u8>) {
        let r_d = reg(rd);
        let h1 = format!("rh({})", reg(rs1));
        let h2 = format!("rh({})", reg(rs2));
        let expr = match op {
            F16Op::Hadd => format!("{r_d} = rhi({h1} + {h2});"),
            F16Op::Hsub => format!("{r_d} = rhi({h1} - {h2});"),
            F16Op::Hmul => format!("{r_d} = rhi({h1} * {h2});"),
            F16Op::Hma => {
                let h3 = format!("rh({})", reg(rs3.unwrap_or(0)));
                format!("{r_d} = rhi(sycl::fma({h1}, {h2}, {h3}));")
            }
        };
        self.line(&expr);
    }

    fn emit_f16_packed(&mut self, op: F16PackedOp, rd: u8, rs1: u8, rs2: u8) {
        let r_d = reg(rd);
        let source1 = reg(rs1);
        let source2 = reg(rs2);
        let expr = match op {
            F16PackedOp::Hadd2 => format!("{r_d} = ri(rf({source1}) + rf({source2}));"),
            F16PackedOp::Hmul2 => format!("{r_d} = ri(rf({source1}) * rf({source2}));"),
            F16PackedOp::Hma2 => format!(
                "{r_d} = ri(sycl::fma(rf({source1}), rf({source2}), rf({})));",
                reg(0)
            ),
        };
        self.line(&expr);
    }

    fn emit_bf16(&mut self, op: Bf16Op, rd: u8, rs1: u8, rs2: u8, rs3: Option<u8>) {
        let rb = |n: u8| format!("bit_cast<float>((r[{n}] & 0xFFFFu) << 16)");
        let r_d = format!("r[{rd}]");
        let b1 = rb(rs1);
        let b2 = rb(rs2);
        let line = match op {
            Bf16Op::Badd => format!("{r_d} = (bit_cast<uint32_t>({b1} + {b2}) >> 16);"),
            Bf16Op::Bsub => format!("{r_d} = (bit_cast<uint32_t>({b1} - {b2}) >> 16);"),
            Bf16Op::Bmul => format!("{r_d} = (bit_cast<uint32_t>({b1} * {b2}) >> 16);"),
            Bf16Op::Bma => {
                let b3 = rb(rs3.unwrap_or(0));
                format!("{r_d} = (bit_cast<uint32_t>(sycl::fma({b1}, {b2}, {b3})) >> 16);")
            }
        };
        self.line(&line);
    }

    fn emit_bf16_packed(&mut self, op: Bf16PackedOp, rd: u8, rs1: u8, rs2: u8) {
        let r_d = reg(rd);
        self.line("{");
        self.indent_level += 1;
        self.line_args(format_args!(
            "float _lo1 = bit_cast<float>((r[{rs1}] & 0xFFFFu) << 16);"
        ));
        self.line_args(format_args!(
            "float _hi1 = bit_cast<float>((r[{rs1}] >> 16) << 16);"
        ));
        self.line_args(format_args!(
            "float _lo2 = bit_cast<float>((r[{rs2}] & 0xFFFFu) << 16);"
        ));
        self.line_args(format_args!(
            "float _hi2 = bit_cast<float>((r[{rs2}] >> 16) << 16);"
        ));
        match op {
            Bf16PackedOp::Badd2 => {
                self.line("float _rlo = _lo1 + _lo2;");
                self.line("float _rhi = _hi1 + _hi2;");
            }
            Bf16PackedOp::Bmul2 => {
                self.line("float _rlo = _lo1 * _lo2;");
                self.line("float _rhi = _hi1 * _hi2;");
            }
            Bf16PackedOp::Bma2 => {
                self.line_args(format_args!(
                    "float _lo3 = bit_cast<float>((r[{}] & 0xFFFFu) << 16);",
                    0
                ));
                self.line_args(format_args!(
                    "float _hi3 = bit_cast<float>((r[{}] >> 16) << 16);",
                    0
                ));
                self.line("float _rlo = sycl::fma(_lo1, _lo2, _lo3);");
                self.line("float _rhi = sycl::fma(_hi1, _hi2, _hi3);");
            }
        }
        self.line_args(format_args!(
            "{r_d} = (bit_cast<uint32_t>(_rlo) >> 16) | ((bit_cast<uint32_t>(_rhi) >> 16) << 16);"
        ));
        self.indent_level = self.indent_level.saturating_sub(1);
        self.line("}");
    }

    fn emit_f64(&mut self, op: F64Op, rd: u8, rs1: u8, rs2: u8, rs3: Option<u8>) {
        let r_d_lo = reg(rd);
        let r_d_hi = reg(rd + 1);
        let s1_lo = reg(rs1);
        let s1_hi = reg(rs1 + 1);
        let s2_lo = reg(rs2);
        let s2_hi = reg(rs2 + 1);
        self.line("{");
        self.indent_level += 1;
        self.line_args(format_args!(
            "uint64_t _ds1 = ((uint64_t){s1_hi} << 32) | (uint64_t){s1_lo};"
        ));
        self.line_args(format_args!(
            "uint64_t _ds2 = ((uint64_t){s2_hi} << 32) | (uint64_t){s2_lo};"
        ));
        let result_expr = match op {
            F64Op::Dadd => "bit_cast<double>(_ds1) + bit_cast<double>(_ds2)".to_string(),
            F64Op::Dsub => "bit_cast<double>(_ds1) - bit_cast<double>(_ds2)".to_string(),
            F64Op::Dmul => "bit_cast<double>(_ds1) * bit_cast<double>(_ds2)".to_string(),
            F64Op::Dma => {
                let s3_lo = reg(rs3.unwrap_or(0));
                let s3_hi = reg(rs3.unwrap_or(0) + 1);
                self.line_args(format_args!(
                    "uint64_t _ds3 = ((uint64_t){s3_hi} << 32) | (uint64_t){s3_lo};"
                ));
                "sycl::fma(bit_cast<double>(_ds1), bit_cast<double>(_ds2), bit_cast<double>(_ds3))"
                    .to_string()
            }
        };
        self.line_args(format_args!(
            "uint64_t _dr = bit_cast<uint64_t>({result_expr});"
        ));
        self.line_args(format_args!("{r_d_lo} = (uint32_t)_dr;"));
        self.line_args(format_args!("{r_d_hi} = (uint32_t)(_dr >> 32);"));
        self.indent_level = self.indent_level.saturating_sub(1);
        self.line("}");
    }

    fn emit_f64_div_sqrt(&mut self, op: F64DivSqrtOp, rd: u8, rs1: u8, rs2: Option<u8>) {
        let r_d_lo = reg(rd);
        let r_d_hi = reg(rd + 1);
        let s1_lo = reg(rs1);
        let s1_hi = reg(rs1 + 1);
        self.line("{");
        self.indent_level += 1;
        self.line_args(format_args!(
            "uint64_t _ds1 = ((uint64_t){s1_hi} << 32) | (uint64_t){s1_lo};"
        ));
        match op {
            F64DivSqrtOp::Ddiv => {
                let s2 = rs2.unwrap_or(0);
                let s2_lo = reg(s2);
                let s2_hi = reg(s2 + 1);
                self.line_args(format_args!(
                    "uint64_t _ds2 = ((uint64_t){s2_hi} << 32) | (uint64_t){s2_lo};"
                ));
                self.line(
                    "uint64_t _dr = bit_cast<uint64_t>(bit_cast<double>(_ds1) / bit_cast<double>(_ds2));",
                );
            }
            F64DivSqrtOp::Dsqrt => {
                self.line("uint64_t _dr = bit_cast<uint64_t>(sycl::sqrt(bit_cast<double>(_ds1)));");
            }
        }
        self.line_args(format_args!("{r_d_lo} = (uint32_t)_dr;"));
        self.line_args(format_args!("{r_d_hi} = (uint32_t)(_dr >> 32);"));
        self.indent_level = self.indent_level.saturating_sub(1);
        self.line("}");
    }

    fn emit_bitop(
        &mut self,
        op: BitOpType,
        rd: u8,
        rs1: u8,
        rs2: Option<u8>,
        rs3: Option<u8>,
        rs4: Option<u8>,
    ) {
        let r_d = reg(rd);
        let source1 = reg(rs1);
        match op {
            BitOpType::Bitcount => {
                self.line_args(format_args!("{r_d} = sycl::popcount({source1});"));
            }
            BitOpType::Bitfind => {
                self.line_args(format_args!(
                    "{r_d} = ({source1} == 0u) ? 0xFFFFFFFFu : sycl::ctz({source1});"
                ));
            }
            BitOpType::Bitrev => {
                self.line("{");
                self.indent_level += 1;
                self.line_args(format_args!("uint32_t _v = {source1};"));
                self.line("_v = ((_v >> 1) & 0x55555555u) | ((_v & 0x55555555u) << 1);");
                self.line("_v = ((_v >> 2) & 0x33333333u) | ((_v & 0x33333333u) << 2);");
                self.line("_v = ((_v >> 4) & 0x0F0F0F0Fu) | ((_v & 0x0F0F0F0Fu) << 4);");
                self.line("_v = ((_v >> 8) & 0x00FF00FFu) | ((_v & 0x00FF00FFu) << 8);");
                self.line_args(format_args!("{r_d} = (_v >> 16) | (_v << 16);"));
                self.indent_level = self.indent_level.saturating_sub(1);
                self.line("}");
            }
            BitOpType::Bfe => {
                let source2 = reg(rs2.unwrap_or(0));
                let source3 = reg(rs3.unwrap_or(0));
                self.line_args(format_args!(
                    "{r_d} = ({source1} >> {source2}) & ((1u << {source3}) - 1u);"
                ));
            }
            BitOpType::Bfi => {
                let source2 = reg(rs2.unwrap_or(0));
                let source3 = reg(rs3.unwrap_or(0));
                let source4 = reg(rs4.unwrap_or(0));
                self.line_args(format_args!(
                    "{r_d} = ({source2} & ~(((1u << {source4}) - 1u) << {source3})) | (({source1} & ((1u << {source4}) - 1u)) << {source3});"
                ));
            }
        }
    }

    fn emit_cmp(&mut self, cast: &str, op: CmpOp, pd: u8, rs1: u8, rs2: u8) {
        let p_d = pred(pd);
        let lhs = if cast.is_empty() {
            reg(rs1)
        } else {
            format!("{cast}{}", reg(rs1))
        };
        let rhs = if cast.is_empty() {
            reg(rs2)
        } else {
            format!("{cast}{}", reg(rs2))
        };
        let cmp_op = match op {
            CmpOp::Eq => "==",
            CmpOp::Ne => "!=",
            CmpOp::Lt => "<",
            CmpOp::Le => "<=",
            CmpOp::Gt => ">",
            CmpOp::Ge => ">=",
            CmpOp::Ord | CmpOp::Unord => {
                self.emit_fcmp(op, pd, rs1, rs2);
                return;
            }
        };
        self.line_args(format_args!("{p_d} = {lhs} {cmp_op} {rhs};"));
    }

    fn emit_fcmp(&mut self, op: CmpOp, pd: u8, rs1: u8, rs2: u8) {
        let p_d = pred(pd);
        let f1 = format!("rf({})", reg(rs1));
        let f2 = format!("rf({})", reg(rs2));
        let expr = match op {
            CmpOp::Eq => format!("{p_d} = {f1} == {f2};"),
            CmpOp::Ne => format!("{p_d} = {f1} != {f2};"),
            CmpOp::Lt => format!("{p_d} = {f1} < {f2};"),
            CmpOp::Le => format!("{p_d} = {f1} <= {f2};"),
            CmpOp::Gt => format!("{p_d} = {f1} > {f2};"),
            CmpOp::Ge => format!("{p_d} = {f1} >= {f2};"),
            CmpOp::Ord => format!("{p_d} = !sycl::isnan({f1}) && !sycl::isnan({f2});"),
            CmpOp::Unord => format!("{p_d} = sycl::isnan({f1}) || sycl::isnan({f2});"),
        };
        self.line(&expr);
    }

    fn emit_cvt(&mut self, cvt_type: CvtType, rd: u8, rs1: u8) {
        let r_d = reg(rd);
        let source1 = reg(rs1);
        let expr = match cvt_type {
            CvtType::F32I32 => format!("{r_d} = (uint32_t)((int32_t)rf({source1}));"),
            CvtType::F32U32 => format!("{r_d} = (uint32_t)rf({source1});"),
            CvtType::I32F32 => format!("{r_d} = ri((float)((int32_t){source1}));"),
            CvtType::U32F32 => format!("{r_d} = ri((float){source1});"),
            CvtType::F32F16 => format!("{r_d} = rhi((half)rf({source1}));"),
            CvtType::F16F32 => format!("{r_d} = ri((float)rh({source1}));"),
            CvtType::F32Bf16 => {
                format!("{r_d} = ri(bit_cast<float>(({source1} & 0xFFFFu) << 16));")
            }
            CvtType::Bf16F32 => {
                format!("{r_d} = (bit_cast<uint32_t>(rf({source1})) >> 16);")
            }
            CvtType::F32F64 => {
                let r_d1 = reg(rd + 1);
                self.line("{");
                self.indent_level += 1;
                self.line_args(format_args!(
                    "uint64_t _dr = bit_cast<uint64_t>((double)rf({source1}));"
                ));
                self.line_args(format_args!("{r_d} = (uint32_t)_dr;"));
                self.line_args(format_args!("{r_d1} = (uint32_t)(_dr >> 32);"));
                self.indent_level = self.indent_level.saturating_sub(1);
                self.line("}");
                return;
            }
            CvtType::F64F32 => {
                let s1_hi = reg(rs1 + 1);
                self.line("{");
                self.indent_level += 1;
                self.line_args(format_args!(
                    "uint64_t _ds = ((uint64_t){s1_hi} << 32) | (uint64_t){source1};"
                ));
                self.line_args(format_args!("{r_d} = ri((float)bit_cast<double>(_ds));"));
                self.indent_level = self.indent_level.saturating_sub(1);
                self.line("}");
                return;
            }
        };
        self.line(&expr);
    }
}
