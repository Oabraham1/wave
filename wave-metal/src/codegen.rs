// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Main code generator. Iterates decoded WAVE instructions from wave-decode and emits
//!
//! equivalent Metal Shading Language (MSL) text. Uses a String buffer with writeln! for
//! emission. Control flow maps directly to MSL if/else/while. Float operations use
//! `rf()`/`ri()` helper functions for bitcasting since WAVE registers are untyped 32-bit.
//! Predicated instructions are wrapped in MSL if-guards. The generator delegates to
//! sub-modules for memory operations, wave intrinsics, and control flow formatting.

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

#[allow(clippy::similar_names)]
impl CodeGenerator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate MSL code for a kernel's instructions.
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
        .unwrap();

        self.output.push_str(&registers::emit_declarations(
            kernel.register_count,
            kernel.local_memory_size,
        ));
        self.line("");

        for instr in instructions {
            self.emit_instruction(instr)?;
        }

        writeln!(self.output, "{}", crate::kernel::emit_kernel_footer()).unwrap();

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
            self.line(item);
        }
    }

    fn indent(&mut self) {
        self.indent_level += 1;
    }

    fn dedent(&mut self) {
        self.indent_level = self.indent_level.saturating_sub(1);
    }

    fn emit_instruction(&mut self, instr: &DecodedInstruction) -> Result<(), CompileError> {
        if instr.predicate != 0 {
            let guard = if instr.predicate_negated {
                format!("if (!{}) {{", pred(instr.predicate))
            } else {
                format!("if ({}) {{", pred(instr.predicate))
            };
            self.line(&guard);
            self.indent();
            self.emit_operation(&instr.operation)?;
            self.dedent();
            self.line("}");
        } else {
            self.emit_operation(&instr.operation)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn emit_operation(&mut self, op: &Operation) -> Result<(), CompileError> {
        match op {
            Operation::Iadd { rd, rs1, rs2 } => {
                self.line(&format!("{} = {} + {};", reg(*rd), reg(*rs1), reg(*rs2)));
            }
            Operation::Isub { rd, rs1, rs2 } => {
                self.line(&format!("{} = {} - {};", reg(*rd), reg(*rs1), reg(*rs2)));
            }
            Operation::Imul { rd, rs1, rs2 } => {
                self.line(&format!("{} = {} * {};", reg(*rd), reg(*rs1), reg(*rs2)));
            }
            Operation::ImulHi { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{} = (uint32_t)(((uint64_t){} * (uint64_t){}) >> 32);",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Imad { rd, rs1, rs2, rs3 } => {
                self.line(&format!(
                    "{} = {} * {} + {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2),
                    reg(*rs3)
                ));
            }
            Operation::Idiv { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{} = (uint32_t)((int32_t){} / (int32_t){});",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Imod { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{} = (uint32_t)((int32_t){} % (int32_t){});",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Ineg { rd, rs1 } => {
                self.line(&format!(
                    "{} = (uint32_t)(-(int32_t){});",
                    reg(*rd),
                    reg(*rs1)
                ));
            }
            Operation::Iabs { rd, rs1 } => {
                self.line(&format!(
                    "{} = (uint32_t)abs((int32_t){});",
                    reg(*rd),
                    reg(*rs1)
                ));
            }
            Operation::Imin { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{} = (uint32_t)min((int32_t){}, (int32_t){});",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Imax { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{} = (uint32_t)max((int32_t){}, (int32_t){});",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Iclamp { rd, rs1, rs2, rs3 } => {
                self.line(&format!(
                    "{} = (uint32_t)clamp((int32_t){}, (int32_t){}, (int32_t){});",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2),
                    reg(*rs3)
                ));
            }

            Operation::Fadd { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{} = ri(rf({}) + rf({}));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Fsub { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{} = ri(rf({}) - rf({}));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Fmul { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{} = ri(rf({}) * rf({}));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Fma { rd, rs1, rs2, rs3 } => {
                self.line(&format!(
                    "{} = ri(fma(rf({}), rf({}), rf({})));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2),
                    reg(*rs3)
                ));
            }
            Operation::Fdiv { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{} = ri(rf({}) / rf({}));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Fneg { rd, rs1 } => {
                self.line(&format!("{} = ri(-rf({}));", reg(*rd), reg(*rs1)));
            }
            Operation::Fabs { rd, rs1 } => {
                self.line(&format!("{} = ri(abs(rf({})));", reg(*rd), reg(*rs1)));
            }
            Operation::Fmin { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{} = ri(min(rf({}), rf({})));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Fmax { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{} = ri(max(rf({}), rf({})));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Fclamp { rd, rs1, rs2, rs3 } => {
                self.line(&format!(
                    "{} = ri(clamp(rf({}), rf({}), rf({})));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2),
                    reg(*rs3)
                ));
            }
            Operation::Fsqrt { rd, rs1 } => {
                self.line(&format!("{} = ri(sqrt(rf({})));", reg(*rd), reg(*rs1)));
            }
            Operation::FUnary { op, rd, rs1 } => {
                self.emit_funary(*op, *rd, *rs1);
            }

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

            Operation::And { rd, rs1, rs2 } => {
                self.line(&format!("{} = {} & {};", reg(*rd), reg(*rs1), reg(*rs2)));
            }
            Operation::Or { rd, rs1, rs2 } => {
                self.line(&format!("{} = {} | {};", reg(*rd), reg(*rs1), reg(*rs2)));
            }
            Operation::Xor { rd, rs1, rs2 } => {
                self.line(&format!("{} = {} ^ {};", reg(*rd), reg(*rs1), reg(*rs2)));
            }
            Operation::Not { rd, rs1 } => {
                self.line(&format!("{} = ~{};", reg(*rd), reg(*rs1)));
            }
            Operation::Shl { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{} = {} << ({} & 0x1Fu);",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Shr { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{} = {} >> ({} & 0x1Fu);",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Sar { rd, rs1, rs2 } => {
                self.line(&format!(
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
                self.line(&format!(
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

            Operation::LocalLoad { width, rd, addr } => {
                let stmts = memory::emit_load("threadgroup", "local_mem", *width, *rd, *addr);
                self.lines(&stmts);
            }
            Operation::LocalStore { width, addr, value } => {
                let stmts = memory::emit_store("threadgroup", "local_mem", *width, *addr, *value);
                self.lines(&stmts);
            }
            Operation::DeviceLoad { width, rd, addr } => {
                let stmts = memory::emit_load("device", "device_mem", *width, *rd, *addr);
                self.lines(&stmts);
            }
            Operation::DeviceStore { width, addr, value } => {
                let stmts = memory::emit_store("device", "device_mem", *width, *addr, *value);
                self.lines(&stmts);
            }

            Operation::LocalAtomic {
                op,
                rd,
                addr,
                value,
            } => {
                let stmts =
                    memory::emit_atomic("threadgroup", "local_mem", *op, *rd, *addr, *value);
                self.lines(&stmts);
            }
            Operation::LocalAtomicCas {
                rd,
                addr,
                expected,
                desired,
            } => {
                let stmts = memory::emit_atomic_cas(
                    "threadgroup",
                    "local_mem",
                    *rd,
                    *addr,
                    *expected,
                    *desired,
                );
                self.lines(&stmts);
            }
            Operation::DeviceAtomic {
                op,
                rd,
                addr,
                value,
                ..
            } => {
                let stmts = memory::emit_atomic("device", "device_mem", *op, *rd, *addr, *value);
                self.lines(&stmts);
            }
            Operation::DeviceAtomicCas {
                rd,
                addr,
                expected,
                desired,
                ..
            } => {
                let stmts = memory::emit_atomic_cas(
                    "device",
                    "device_mem",
                    *rd,
                    *addr,
                    *expected,
                    *desired,
                );
                self.lines(&stmts);
            }

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

            Operation::If { ps } => {
                self.line(&control_flow::emit_if(*ps));
                self.indent();
            }
            Operation::Else => {
                self.dedent();
                self.line(control_flow::emit_else());
                self.indent();
            }
            Operation::Endif => {
                self.dedent();
                self.line(control_flow::emit_endif());
            }
            Operation::Loop => {
                self.line(control_flow::emit_loop());
                self.indent();
            }
            Operation::Break { ps } => {
                self.line(&control_flow::emit_break(*ps));
            }
            Operation::Continue { ps } => {
                self.line(&control_flow::emit_continue(*ps));
            }
            Operation::Endloop => {
                self.dedent();
                self.line(control_flow::emit_endloop());
            }
            Operation::Call { .. } => {
                return Err(CompileError::UnsupportedOperation(
                    "call instructions require function extraction which is not supported in the Metal backend".to_string(),
                ));
            }

            Operation::Return | Operation::Halt => {
                self.line("return;");
            }
            Operation::Barrier => {
                self.line(
                    "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);",
                );
            }
            Operation::FenceAcquire { .. }
            | Operation::FenceRelease { .. }
            | Operation::FenceAcqRel { .. } => {
                self.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
            }
            Operation::Wait | Operation::Nop => {}

            Operation::Mov { rd, rs1 } => {
                self.line(&format!("{} = {};", reg(*rd), reg(*rs1)));
            }
            Operation::MovImm { rd, imm } => {
                self.line(&format!("{} = {imm}u;", reg(*rd)));
            }
            Operation::MovSr { rd, sr_index } => {
                self.line(&format!(
                    "{} = (uint32_t){};",
                    reg(*rd),
                    registers::special_reg_expr(*sr_index)
                ));
            }

            Operation::MmaLoadA { rd: _, rs1, rs2 } => {
                self.line("{");
                self.line(&format!("    uint _addr = {};", reg(*rs1)));
                self.line(&format!("    uint _stride = {};", reg(*rs2)));
                self.line("    for (uint _r = 0; _r < 4; _r++) {");
                self.line("        for (uint _c = 0; _c < 4; _c++) {");
                self.line("            _mma_a[_r * 4 + _c] = rf(*(threadgroup uint32_t*)(local_mem + _addr + _r * _stride + _c * 4));");
                self.line("        }");
                self.line("    }");
                self.line("}");
            }
            Operation::MmaLoadB { rd: _, rs1, rs2 } => {
                self.line("{");
                self.line(&format!("    uint _addr = {};", reg(*rs1)));
                self.line(&format!("    uint _stride = {};", reg(*rs2)));
                self.line("    for (uint _r = 0; _r < 4; _r++) {");
                self.line("        for (uint _c = 0; _c < 4; _c++) {");
                self.line("            _mma_b[_r * 4 + _c] = rf(*(threadgroup uint32_t*)(local_mem + _addr + _r * _stride + _c * 4));");
                self.line("        }");
                self.line("    }");
                self.line("}");
            }
            Operation::MmaStoreC { rd, rs1, rs2: _ } => {
                self.line("{");
                self.line(&format!("    uint _addr = {};", reg(*rd)));
                self.line(&format!("    uint _stride = {};", reg(*rs1)));
                self.line("    for (uint _r = 0; _r < 4; _r++) {");
                self.line("        for (uint _c = 0; _c < 4; _c++) {");
                self.line("            *(threadgroup uint32_t*)(local_mem + _addr + _r * _stride + _c * 4) = ri(_mma_c[_r * 4 + _c]);");
                self.line("        }");
                self.line("    }");
                self.line("}");
            }
            Operation::MmaCompute { .. } => {
                self.line("for (uint _i = 0; _i < 4; _i++) {");
                self.line("    for (uint _j = 0; _j < 4; _j++) {");
                self.line("        for (uint _k = 0; _k < 4; _k++) {");
                self.line("            _mma_c[_i * 4 + _j] = fma(_mma_a[_i * 4 + _k], _mma_b[_k * 4 + _j], _mma_c[_i * 4 + _j]);");
                self.line("        }");
                self.line("    }");
                self.line("}");
            }

            Operation::Unknown { opcode, .. } => {
                return Err(CompileError::UnsupportedOperation(format!(
                    "unknown opcode 0x{opcode:02x}"
                )));
            }
        }
        Ok(())
    }

    fn emit_funary(&mut self, op: FUnaryOp, rd: u8, rs1: u8) {
        let r_d = reg(rd);
        let r_s1 = reg(rs1);
        let expr = match op {
            FUnaryOp::Frsqrt => format!("{r_d} = ri(rsqrt(rf({r_s1})));"),
            FUnaryOp::Frcp => format!("{r_d} = ri(1.0f / rf({r_s1}));"),
            FUnaryOp::Ffloor => format!("{r_d} = ri(floor(rf({r_s1})));"),
            FUnaryOp::Fceil => format!("{r_d} = ri(ceil(rf({r_s1})));"),
            FUnaryOp::Fround => format!("{r_d} = ri(rint(rf({r_s1})));"),
            FUnaryOp::Ftrunc => format!("{r_d} = ri(trunc(rf({r_s1})));"),
            FUnaryOp::Ffract => format!("{r_d} = ri(fract(rf({r_s1})));"),
            FUnaryOp::Fsat => format!("{r_d} = ri(saturate(rf({r_s1})));"),
            FUnaryOp::Fsin => format!("{r_d} = ri(sin(rf({r_s1})));"),
            FUnaryOp::Fcos => format!("{r_d} = ri(cos(rf({r_s1})));"),
            FUnaryOp::Fexp2 => format!("{r_d} = ri(exp2(rf({r_s1})));"),
            FUnaryOp::Flog2 => format!("{r_d} = ri(log2(rf({r_s1})));"),
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
                format!("{r_d} = rhi(fma({h1}, {h2}, {h3}));")
            }
        };
        self.line(&expr);
    }

    fn emit_f16_packed(&mut self, op: F16PackedOp, rd: u8, rs1: u8, rs2: u8, rs3: Option<u8>) {
        let r_d = reg(rd);
        let h1 = format!("rh2({})", reg(rs1));
        let h2 = format!("rh2({})", reg(rs2));

        let expr = match op {
            F16PackedOp::Hadd2 => format!("{r_d} = rh2i({h1} + {h2});"),
            F16PackedOp::Hmul2 => format!("{r_d} = rh2i({h1} * {h2});"),
            F16PackedOp::Hma2 => {
                let h3 = format!("rh2({})", reg(rs3.unwrap_or(0)));
                format!("{r_d} = rh2i(fma({h1}, {h2}, {h3}));")
            }
        };
        self.line(&expr);
    }

    fn emit_bf16(&mut self, op: Bf16Op, rd: u8, rs1: u8, rs2: u8, rs3: Option<u8>) {
        let r_d = reg(rd);
        let b1 = format!("as_type<float>(({} & 0xFFFFu) << 16)", reg(rs1));
        let b2 = format!("as_type<float>(({} & 0xFFFFu) << 16)", reg(rs2));

        let expr = match op {
            Bf16Op::Badd => format!("{r_d} = (as_type<uint>({b1} + {b2}) >> 16);"),
            Bf16Op::Bsub => format!("{r_d} = (as_type<uint>({b1} - {b2}) >> 16);"),
            Bf16Op::Bmul => format!("{r_d} = (as_type<uint>({b1} * {b2}) >> 16);"),
            Bf16Op::Bma => {
                let b3 = format!(
                    "as_type<float>(({} & 0xFFFFu) << 16)",
                    reg(rs3.unwrap_or(0))
                );
                format!("{r_d} = (as_type<uint>(fma({b1}, {b2}, {b3})) >> 16);")
            }
        };
        self.line(&expr);
    }

    fn emit_bf16_packed(&mut self, op: Bf16PackedOp, rd: u8, rs1: u8, rs2: u8, rs3: Option<u8>) {
        let r_d = reg(rd);
        let r_s1 = reg(rs1);
        let r_s2 = reg(rs2);

        self.line("{");
        self.indent();
        self.line(&format!(
            "float b1_lo = as_type<float>(({r_s1} & 0xFFFFu) << 16);"
        ));
        self.line(&format!(
            "float b1_hi = as_type<float>({r_s1} & 0xFFFF0000u);"
        ));
        self.line(&format!(
            "float b2_lo = as_type<float>(({r_s2} & 0xFFFFu) << 16);"
        ));
        self.line(&format!(
            "float b2_hi = as_type<float>({r_s2} & 0xFFFF0000u);"
        ));

        match op {
            Bf16PackedOp::Badd2 => {
                self.line("uint lo = (as_type<uint>(b1_lo + b2_lo) >> 16);");
                self.line("uint hi = (as_type<uint>(b1_hi + b2_hi) & 0xFFFF0000u);");
            }
            Bf16PackedOp::Bmul2 => {
                self.line("uint lo = (as_type<uint>(b1_lo * b2_lo) >> 16);");
                self.line("uint hi = (as_type<uint>(b1_hi * b2_hi) & 0xFFFF0000u);");
            }
            Bf16PackedOp::Bma2 => {
                let r_s3 = reg(rs3.unwrap_or(0));
                self.line(&format!(
                    "float b3_lo = as_type<float>(({r_s3} & 0xFFFFu) << 16);"
                ));
                self.line(&format!(
                    "float b3_hi = as_type<float>({r_s3} & 0xFFFF0000u);"
                ));
                self.line("uint lo = (as_type<uint>(fma(b1_lo, b2_lo, b3_lo)) >> 16);");
                self.line("uint hi = (as_type<uint>(fma(b1_hi, b2_hi, b3_hi)) & 0xFFFF0000u);");
            }
        }

        self.line(&format!("{r_d} = hi | lo;"));
        self.dedent();
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
        self.indent();
        self.line(&format!(
            "ulong d_s1 = ((ulong){s1_hi} << 32) | (ulong){s1_lo};"
        ));
        self.line(&format!(
            "ulong d_s2 = ((ulong){s2_hi} << 32) | (ulong){s2_lo};"
        ));

        let result_expr = match op {
            F64Op::Dadd => "as_type<double>(d_s1) + as_type<double>(d_s2)".to_string(),
            F64Op::Dsub => "as_type<double>(d_s1) - as_type<double>(d_s2)".to_string(),
            F64Op::Dmul => "as_type<double>(d_s1) * as_type<double>(d_s2)".to_string(),
            F64Op::Dma => {
                let s3_lo = reg(rs3.unwrap_or(0));
                let s3_hi = reg(rs3.unwrap_or(0) + 1);
                self.line(&format!(
                    "ulong d_s3 = ((ulong){s3_hi} << 32) | (ulong){s3_lo};"
                ));
                "fma(as_type<double>(d_s1), as_type<double>(d_s2), as_type<double>(d_s3))"
                    .to_string()
            }
        };

        self.line(&format!("ulong d_result = as_type<ulong>({result_expr});"));
        self.line(&format!("{r_d_lo} = (uint32_t)d_result;"));
        self.line(&format!("{r_d_hi} = (uint32_t)(d_result >> 32);"));
        self.dedent();
        self.line("}");
    }

    fn emit_f64_div_sqrt(&mut self, op: F64DivSqrtOp, rd: u8, rs1: u8, rs2: Option<u8>) {
        let r_d_lo = reg(rd);
        let r_d_hi = reg(rd + 1);
        let s1_lo = reg(rs1);
        let s1_hi = reg(rs1 + 1);

        self.line("{");
        self.indent();
        self.line(&format!(
            "ulong d_s1 = ((ulong){s1_hi} << 32) | (ulong){s1_lo};"
        ));

        match op {
            F64DivSqrtOp::Ddiv => {
                let s2 = rs2.unwrap_or(0);
                let s2_lo = reg(s2);
                let s2_hi = reg(s2 + 1);
                self.line(&format!(
                    "ulong d_s2 = ((ulong){s2_hi} << 32) | (ulong){s2_lo};"
                ));
                self.line(
                    "ulong d_result = as_type<ulong>(as_type<double>(d_s1) / as_type<double>(d_s2));",
                );
            }
            F64DivSqrtOp::Dsqrt => {
                self.line("ulong d_result = as_type<ulong>(sqrt(as_type<double>(d_s1)));");
            }
        }

        self.line(&format!("{r_d_lo} = (uint32_t)d_result;"));
        self.line(&format!("{r_d_hi} = (uint32_t)(d_result >> 32);"));
        self.dedent();
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
        let r_s1 = reg(rs1);

        match op {
            BitOpType::Bitcount => {
                self.line(&format!("{r_d} = popcount({r_s1});"));
            }
            BitOpType::Bitfind => {
                self.line(&format!(
                    "{r_d} = ({r_s1} == 0u) ? 0xFFFFFFFFu : ctz({r_s1});"
                ));
            }
            BitOpType::Bitrev => {
                self.line(&format!("{r_d} = reverse_bits({r_s1});"));
            }
            BitOpType::Bfe => {
                let r_s2 = reg(rs2.unwrap_or(0));
                let r_s3 = reg(rs3.unwrap_or(0));
                self.line(&format!("{r_d} = extract_bits({r_s1}, {r_s2}, {r_s3});"));
            }
            BitOpType::Bfi => {
                let r_s2 = reg(rs2.unwrap_or(0));
                let r_s3 = reg(rs3.unwrap_or(0));
                let r_s4 = reg(rs4.unwrap_or(0));
                self.line(&format!(
                    "{r_d} = insert_bits({r_s1}, {r_s2}, {r_s3}, {r_s4});"
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

        self.line(&format!("{p_d} = {lhs} {cmp_op} {rhs};"));
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
        let r_s1 = reg(rs1);

        let expr = match cvt_type {
            CvtType::F32I32 => format!("{r_d} = (uint32_t)((int32_t)rf({r_s1}));"),
            CvtType::F32U32 => format!("{r_d} = (uint32_t)rf({r_s1});"),
            CvtType::I32F32 => format!("{r_d} = ri((float)((int32_t){r_s1}));"),
            CvtType::U32F32 => format!("{r_d} = ri((float){r_s1});"),
            CvtType::F32F16 => {
                format!("{r_d} = ri((float)as_type<half>((ushort)({r_s1} & 0xFFFFu)));")
            }
            CvtType::F16F32 => format!("{r_d} = (uint32_t)as_type<ushort>(half(rf({r_s1})));"),
            CvtType::F32Bf16 => {
                format!("{r_d} = (as_type<uint>(rf({r_s1})) >> 16);")
            }
            CvtType::Bf16F32 => {
                format!("{r_d} = ri(as_type<float>(({r_s1} & 0xFFFFu) << 16));")
            }
            CvtType::F32F64 => {
                let r_d1 = reg(rd + 1);
                self.line("{");
                self.indent();
                self.line(&format!(
                    "ulong d_result = as_type<ulong>((double)rf({r_s1}));"
                ));
                self.line(&format!("{r_d} = (uint32_t)d_result;"));
                self.line(&format!("{r_d1} = (uint32_t)(d_result >> 32);"));
                self.dedent();
                self.line("}");
                return;
            }
            CvtType::F64F32 => {
                let s1_hi = reg(rs1 + 1);
                self.line("{");
                self.indent();
                self.line(&format!(
                    "ulong d_s1 = ((ulong){s1_hi} << 32) | (ulong){r_s1};"
                ));
                self.line(&format!("{r_d} = ri((float)as_type<double>(d_s1));"));
                self.dedent();
                self.line("}");
                return;
            }
        };
        self.line(&expr);
    }
}
