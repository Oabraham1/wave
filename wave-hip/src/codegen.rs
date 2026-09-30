// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Translate decoded WAVE operations into HIP C++.

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
            indent_level: 1,
        }
    }
}

impl CodeGenerator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate HIP code for a kernel's instructions.
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

        writeln!(
            self.output,
            "{}",
            crate::kernel::emit_kernel_signature(&kernel.name)
        )
        .expect("writing to String cannot fail");

        self.output.push_str(&registers::emit_declarations(
            kernel.register_count,
            kernel.local_memory_size > 0,
        ));
        self.line("");

        for instr in instructions {
            self.emit_instruction(instr)?;
        }

        writeln!(self.output, "{}", crate::kernel::emit_kernel_footer())
            .expect("writing to String cannot fail");

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
                    "{} = (uint32_t)abs((int32_t){});",
                    reg(*rd),
                    reg(*rs1)
                ));
            }
            Operation::Imin { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = (uint32_t)min((int32_t){}, (int32_t){});",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Imax { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = (uint32_t)max((int32_t){}, (int32_t){});",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Iclamp { rd, rs1, rs2, rs3 } => {
                self.line_args(format_args!(
                    "{} = (uint32_t)clamp((int32_t){}, (int32_t){}, (int32_t){});",
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
                    "{} = ri(fmaf(rf({}), rf({}), rf({})));",
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
                self.line_args(format_args!("{} = ri(fabsf(rf({})));", reg(*rd), reg(*rs1)));
            }
            Operation::Fmin { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = ri(fminf(rf({}), rf({})));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Fmax { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = ri(fmaxf(rf({}), rf({})));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Fclamp { rd, rs1, rs2, rs3 } => {
                self.line_args(format_args!(
                    "{} = ri(fminf(fmaxf(rf({}), rf({})), rf({})));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2),
                    reg(*rs3)
                ));
            }
            Operation::Fsqrt { rd, rs1 } => {
                self.line_args(format_args!("{} = ri(sqrtf(rf({})));", reg(*rd), reg(*rs1)));
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
                op,
                rd,
                rs1,
                rs2,
                rs3,
            } => {
                self.emit_f16_packed(*op, *rd, *rs1, *rs2, *rs3);
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
                op,
                rd,
                rs1,
                rs2,
                rs3,
            } => {
                self.emit_bf16_packed(*op, *rd, *rs1, *rs2, *rs3);
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
                let stmts = memory::emit_load("local_mem", *width, *rd, *addr);
                self.lines(&stmts);
            }
            Operation::LocalStore { width, addr, value } => {
                let stmts = memory::emit_store("local_mem", *width, *addr, *value);
                self.lines(&stmts);
            }
            Operation::DeviceLoad { width, rd, addr } => {
                let stmts = memory::emit_load("device_mem", *width, *rd, *addr);
                self.lines(&stmts);
            }
            Operation::DeviceStore { width, addr, value } => {
                let stmts = memory::emit_store("device_mem", *width, *addr, *value);
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
                let stmts = memory::emit_atomic("local_mem", *op, *rd, *addr, *value);
                self.lines(&stmts);
            }
            Operation::LocalAtomicCas {
                rd,
                addr,
                expected,
                desired,
            } => {
                let stmts = memory::emit_atomic_cas("local_mem", *rd, *addr, *expected, *desired);
                self.lines(&stmts);
            }
            Operation::DeviceAtomic {
                op,
                rd,
                addr,
                value,
                ..
            } => {
                let stmts = memory::emit_atomic("device_mem", *op, *rd, *addr, *value);
                self.lines(&stmts);
            }
            Operation::DeviceAtomicCas {
                rd,
                addr,
                expected,
                desired,
                ..
            } => {
                let stmts = memory::emit_atomic_cas("device_mem", *rd, *addr, *expected, *desired);
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
                let stmts = intrinsics::emit_wave_reduce(*op, *rd, *rs1);
                self.lines(&stmts);
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
                self.line("            _mma_a[_r * 4 + _c] = rf(*(uint32_t*)(local_mem + _addr + _r * _stride + _c * 4));");
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
                self.line("            _mma_b[_r * 4 + _c] = rf(*(uint32_t*)(local_mem + _addr + _r * _stride + _c * 4));");
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
                self.line("            *(uint32_t*)(local_mem + _addr + _r * _stride + _c * 4) = ri(_mma_c[_r * 4 + _c]);");
                self.line("        }");
                self.line("    }");
                self.line("}");
            }
            Operation::MmaCompute { .. } => {
                self.line("for (int _i = 0; _i < 4; _i++) {");
                self.line("    for (int _j = 0; _j < 4; _j++) {");
                self.line("        for (int _k = 0; _k < 4; _k++) {");
                self.line("            _mma_c[_i * 4 + _j] = fmaf(_mma_a[_i * 4 + _k], _mma_b[_k * 4 + _j], _mma_c[_i * 4 + _j]);");
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
                    "call instructions are not supported in the HIP backend".to_string(),
                ));
            }
            Operation::Return | Operation::Halt => {
                self.line("return;");
            }
            Operation::Barrier => {
                self.line("__syncthreads();");
            }
            Operation::FenceAcquire { scope }
            | Operation::FenceRelease { scope }
            | Operation::FenceAcqRel { scope } => {
                let fence = match scope {
                    wave_decode::Scope::Wave | wave_decode::Scope::Workgroup => {
                        "__threadfence_block();"
                    }
                    wave_decode::Scope::Device => "__threadfence();",
                    wave_decode::Scope::System => "__threadfence_system();",
                };
                self.line(fence);
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

    fn emit_funary(&mut self, op: FUnaryOp, rd: u8, rs1: u8) {
        let r_d = reg(rd);
        let source1 = reg(rs1);
        let expr = match op {
            FUnaryOp::Frsqrt => format!("{r_d} = ri(rsqrtf(rf({source1})));"),
            FUnaryOp::Frcp => format!("{r_d} = ri(1.0f / rf({source1}));"),
            FUnaryOp::Ffloor => format!("{r_d} = ri(floorf(rf({source1})));"),
            FUnaryOp::Fceil => format!("{r_d} = ri(ceilf(rf({source1})));"),
            FUnaryOp::Fround => format!("{r_d} = ri(rintf(rf({source1})));"),
            FUnaryOp::Ftrunc => format!("{r_d} = ri(truncf(rf({source1})));"),
            FUnaryOp::Ffract => format!("{r_d} = ri(rf({source1}) - floorf(rf({source1})));"),
            FUnaryOp::Fsat => format!("{r_d} = ri(fminf(fmaxf(rf({source1}), 0.0f), 1.0f));"),
            FUnaryOp::Fsin => format!("{r_d} = ri(__sinf(rf({source1})));"),
            FUnaryOp::Fcos => format!("{r_d} = ri(__cosf(rf({source1})));"),
            FUnaryOp::Fexp2 => format!("{r_d} = ri(exp2f(rf({source1})));"),
            FUnaryOp::Flog2 => format!("{r_d} = ri(log2f(rf({source1})));"),
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
                format!("{r_d} = rhi(__hfma({h1}, {h2}, {h3}));")
            }
        };
        self.line(&expr);
    }

    fn emit_f16_packed(&mut self, op: F16PackedOp, rd: u8, rs1: u8, rs2: u8, rs3: Option<u8>) {
        let r_d = reg(rd);
        let s1 = reg(rs1);
        let s2 = reg(rs2);

        let expr = match op {
            F16PackedOp::Hadd2 => format!("{r_d} = ri(__hadd2(rf({s1}), rf({s2})));"),
            F16PackedOp::Hmul2 => format!("{r_d} = ri(__hmul2(rf({s1}), rf({s2})));"),
            F16PackedOp::Hma2 => {
                let s3 = reg(rs3.unwrap_or(0));
                format!("{r_d} = ri(__hfma2(rf({s1}), rf({s2}), rf({s3})));")
            }
        };
        self.line(&expr);
    }

    fn emit_bf16(&mut self, op: Bf16Op, rd: u8, rs1: u8, rs2: u8, rs3: Option<u8>) {
        let rb = |n: u8| format!("__uint_as_float((r[{n}] & 0xFFFFu) << 16)");
        let r_d = format!("r[{rd}]");
        let b1 = rb(rs1);
        let b2 = rb(rs2);
        let line = match op {
            Bf16Op::Badd => format!("{r_d} = (__float_as_uint({b1} + {b2}) >> 16);"),
            Bf16Op::Bsub => format!("{r_d} = (__float_as_uint({b1} - {b2}) >> 16);"),
            Bf16Op::Bmul => format!("{r_d} = (__float_as_uint({b1} * {b2}) >> 16);"),
            Bf16Op::Bma => {
                let b3 = rb(rs3.unwrap_or(0));
                format!("{r_d} = (__float_as_uint(fmaf({b1}, {b2}, {b3})) >> 16);")
            }
        };
        self.line(&line);
    }

    fn emit_bf16_packed(&mut self, op: Bf16PackedOp, rd: u8, rs1: u8, rs2: u8, rs3: Option<u8>) {
        let r_d = format!("r[{rd}]");
        self.line("{");
        self.indent_level += 1;
        self.line_args(format_args!(
            "float _b1lo = __uint_as_float((r[{rs1}] & 0xFFFFu) << 16);"
        ));
        self.line_args(format_args!(
            "float _b1hi = __uint_as_float(r[{rs1}] & 0xFFFF0000u);"
        ));
        self.line_args(format_args!(
            "float _b2lo = __uint_as_float((r[{rs2}] & 0xFFFFu) << 16);"
        ));
        self.line_args(format_args!(
            "float _b2hi = __uint_as_float(r[{rs2}] & 0xFFFF0000u);"
        ));
        match op {
            Bf16PackedOp::Badd2 => {
                self.line("float _rlo = _b1lo + _b2lo;");
                self.line("float _rhi = _b1hi + _b2hi;");
            }
            Bf16PackedOp::Bmul2 => {
                self.line("float _rlo = _b1lo * _b2lo;");
                self.line("float _rhi = _b1hi * _b2hi;");
            }
            Bf16PackedOp::Bma2 => {
                let s3 = rs3.unwrap_or(0);
                self.line_args(format_args!(
                    "float _b3lo = __uint_as_float((r[{s3}] & 0xFFFFu) << 16);"
                ));
                self.line_args(format_args!(
                    "float _b3hi = __uint_as_float(r[{s3}] & 0xFFFF0000u);"
                ));
                self.line("float _rlo = fmaf(_b1lo, _b2lo, _b3lo);");
                self.line("float _rhi = fmaf(_b1hi, _b2hi, _b3hi);");
            }
        }
        self.line_args(format_args!(
            "{r_d} = (__float_as_uint(_rlo) >> 16) | (__float_as_uint(_rhi) & 0xFFFF0000u);"
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
            F64Op::Dadd => "__longlong_as_double(_ds1) + __longlong_as_double(_ds2)".to_string(),
            F64Op::Dsub => "__longlong_as_double(_ds1) - __longlong_as_double(_ds2)".to_string(),
            F64Op::Dmul => "__longlong_as_double(_ds1) * __longlong_as_double(_ds2)".to_string(),
            F64Op::Dma => {
                let s3_lo = reg(rs3.unwrap_or(0));
                let s3_hi = reg(rs3.unwrap_or(0) + 1);
                self.line_args(format_args!(
                    "uint64_t _ds3 = ((uint64_t){s3_hi} << 32) | (uint64_t){s3_lo};"
                ));
                "fma(__longlong_as_double(_ds1), __longlong_as_double(_ds2), __longlong_as_double(_ds3))".to_string()
            }
        };

        self.line_args(format_args!(
            "uint64_t _dr = __double_as_longlong({result_expr});"
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
                    "uint64_t _dr = __double_as_longlong(__longlong_as_double(_ds1) / __longlong_as_double(_ds2));",
                );
            }
            F64DivSqrtOp::Dsqrt => {
                self.line("uint64_t _dr = __double_as_longlong(sqrt(__longlong_as_double(_ds1)));");
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
                self.line_args(format_args!("{r_d} = __popc({source1});"));
            }
            BitOpType::Bitfind => {
                self.line_args(format_args!(
                    "{r_d} = ({source1} == 0u) ? 0xFFFFFFFFu : ((uint32_t)__ffs((int){source1}) - 1u);"
                ));
            }
            BitOpType::Bitrev => {
                self.line_args(format_args!("{r_d} = __brev({source1});"));
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
            CmpOp::Ord => format!("{p_d} = !isnan({f1}) && !isnan({f2});"),
            CmpOp::Unord => format!("{p_d} = isnan({f1}) || isnan({f2});"),
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
            CvtType::F32F16 => format!("{r_d} = rhi(__float2half(rf({source1})));"),
            CvtType::F16F32 => format!("{r_d} = ri(__half2float(rh({source1})));"),
            CvtType::Bf16F32 => {
                format!("{r_d} = __uint_as_float(({source1} & 0xFFFFu) << 16);")
            }
            CvtType::F32Bf16 => {
                format!("{r_d} = (__float_as_uint(__uint_as_float({source1})) >> 16);")
            }
            CvtType::F32F64 => {
                let r_d1 = reg(rd + 1);
                self.line("{");
                self.indent_level += 1;
                self.line_args(format_args!(
                    "uint64_t _dr = __double_as_longlong((double)rf({source1}));"
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
                self.line_args(format_args!(
                    "{r_d} = ri((float)__longlong_as_double(_ds));"
                ));
                self.indent_level = self.indent_level.saturating_sub(1);
                self.line("}");
                return;
            }
        };
        self.line(&expr);
    }
}
