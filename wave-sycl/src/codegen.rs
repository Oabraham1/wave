// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Main code generator for the SYCL backend. Iterates decoded WAVE instructions and
//!
//! emits equivalent SYCL C++ inside a kernel lambda. The kernel body lives at indent
//! level 4 (function > submit > `parallel_for` > lambda). Integer and bitwise operations
//! use standard C++ operators. Float operations use `rf()`/`ri()` helpers wrapping
//! `sycl::bit_cast`. Sub-group operations use SYCL 2020 standard group functions.
//! Atomics use `sycl::atomic_ref`. Local memory is accessed via the lm pointer
//! obtained from a `sycl::local_accessor`.

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
            indent_level: 4,
        }
    }
}

#[allow(clippy::similar_names)]
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

        self.output
            .push_str(&registers::emit_declarations(kernel.register_count, kernel.local_memory_size > 0));
        self.line("");

        for instr in instructions {
            self.emit_instruction(instr)?;
        }

        self.output.push_str(crate::kernel::emit_launch_end());

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
                    "{} = (uint32_t)sycl::abs((int32_t){});",
                    reg(*rd),
                    reg(*rs1)
                ));
            }
            Operation::Imin { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{} = (uint32_t)sycl::min((int32_t){}, (int32_t){});",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Imax { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{} = (uint32_t)sycl::max((int32_t){}, (int32_t){});",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Iclamp { rd, rs1, rs2, rs3 } => {
                self.line(&format!(
                    "{} = (uint32_t)sycl::clamp((int32_t){}, (int32_t){}, (int32_t){});",
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
                    "{} = ri(sycl::fma(rf({}), rf({}), rf({})));",
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
                self.line(&format!(
                    "{} = ri(sycl::fabs(rf({})));",
                    reg(*rd),
                    reg(*rs1)
                ));
            }
            Operation::Fmin { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{} = ri(sycl::fmin(rf({}), rf({})));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Fmax { rd, rs1, rs2 } => {
                self.line(&format!(
                    "{} = ri(sycl::fmax(rf({}), rf({})));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Fclamp { rd, rs1, rs2, rs3 } => {
                self.line(&format!(
                    "{} = ri(sycl::clamp(rf({}), rf({}), rf({})));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2),
                    reg(*rs3)
                ));
            }
            Operation::Fsqrt { rd, rs1 } => {
                self.line(&format!(
                    "{} = ri(sycl::sqrt(rf({})));",
                    reg(*rd),
                    reg(*rs1)
                ));
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
                self.line(&format!("    uint32_t _addr = {};", reg(*rs1)));
                self.line(&format!("    uint32_t _stride = {};", reg(*rs2)));
                self.line("    for (int _r = 0; _r < 4; _r++) {");
                self.line("        for (int _c = 0; _c < 4; _c++) {");
                self.line("            _mma_a[_r * 4 + _c] = rf(*(uint32_t*)(lm + _addr + _r * _stride + _c * 4));");
                self.line("        }");
                self.line("    }");
                self.line("}");
            }
            Operation::MmaLoadB { rd: _, rs1, rs2 } => {
                self.line("{");
                self.line(&format!("    uint32_t _addr = {};", reg(*rs1)));
                self.line(&format!("    uint32_t _stride = {};", reg(*rs2)));
                self.line("    for (int _r = 0; _r < 4; _r++) {");
                self.line("        for (int _c = 0; _c < 4; _c++) {");
                self.line("            _mma_b[_r * 4 + _c] = rf(*(uint32_t*)(lm + _addr + _r * _stride + _c * 4));");
                self.line("        }");
                self.line("    }");
                self.line("}");
            }
            Operation::MmaStoreC { rd, rs1, rs2: _ } => {
                self.line("{");
                self.line(&format!("    uint32_t _addr = {};", reg(*rd)));
                self.line(&format!("    uint32_t _stride = {};", reg(*rs1)));
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

            Operation::Unknown { opcode, .. } => {
                return Err(CompileError::UnsupportedOperation(format!(
                    "unknown opcode 0x{opcode:02x}"
                )));
            }
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
        self.line(&format!("atomic_fence({order}, {sycl_scope});"));
    }

    fn emit_funary(&mut self, op: FUnaryOp, rd: u8, rs1: u8) {
        let r_d = reg(rd);
        let r_s1 = reg(rs1);
        let expr = match op {
            FUnaryOp::Frsqrt => format!("{r_d} = ri(sycl::rsqrt(rf({r_s1})));"),
            FUnaryOp::Frcp => format!("{r_d} = ri(1.0f / rf({r_s1}));"),
            FUnaryOp::Ffloor => format!("{r_d} = ri(sycl::floor(rf({r_s1})));"),
            FUnaryOp::Fceil => format!("{r_d} = ri(sycl::ceil(rf({r_s1})));"),
            FUnaryOp::Fround => format!("{r_d} = ri(sycl::rint(rf({r_s1})));"),
            FUnaryOp::Ftrunc => format!("{r_d} = ri(sycl::trunc(rf({r_s1})));"),
            FUnaryOp::Ffract => format!("{r_d} = ri(rf({r_s1}) - sycl::floor(rf({r_s1})));"),
            FUnaryOp::Fsat => format!("{r_d} = ri(sycl::clamp(rf({r_s1}), 0.0f, 1.0f));"),
            FUnaryOp::Fsin => format!("{r_d} = ri(sycl::sin(rf({r_s1})));"),
            FUnaryOp::Fcos => format!("{r_d} = ri(sycl::cos(rf({r_s1})));"),
            FUnaryOp::Fexp2 => format!("{r_d} = ri(sycl::exp2(rf({r_s1})));"),
            FUnaryOp::Flog2 => format!("{r_d} = ri(sycl::log2(rf({r_s1})));"),
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
        let r_s1 = reg(rs1);
        let r_s2 = reg(rs2);
        let expr = match op {
            F16PackedOp::Hadd2 => format!("{r_d} = ri(rf({r_s1}) + rf({r_s2}));"),
            F16PackedOp::Hmul2 => format!("{r_d} = ri(rf({r_s1}) * rf({r_s2}));"),
            F16PackedOp::Hma2 => format!(
                "{r_d} = ri(sycl::fma(rf({r_s1}), rf({r_s2}), rf({})));",
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
        self.line(&format!(
            "float _lo1 = bit_cast<float>((r[{rs1}] & 0xFFFFu) << 16);"
        ));
        self.line(&format!(
            "float _hi1 = bit_cast<float>((r[{rs1}] >> 16) << 16);"
        ));
        self.line(&format!(
            "float _lo2 = bit_cast<float>((r[{rs2}] & 0xFFFFu) << 16);"
        ));
        self.line(&format!(
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
                self.line(&format!(
                    "float _lo3 = bit_cast<float>((r[{}] & 0xFFFFu) << 16);",
                    0
                ));
                self.line(&format!(
                    "float _hi3 = bit_cast<float>((r[{}] >> 16) << 16);",
                    0
                ));
                self.line("float _rlo = sycl::fma(_lo1, _lo2, _lo3);");
                self.line("float _rhi = sycl::fma(_hi1, _hi2, _hi3);");
            }
        }
        self.line(&format!(
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
        self.line(&format!(
            "uint64_t _ds1 = ((uint64_t){s1_hi} << 32) | (uint64_t){s1_lo};"
        ));
        self.line(&format!(
            "uint64_t _ds2 = ((uint64_t){s2_hi} << 32) | (uint64_t){s2_lo};"
        ));
        let result_expr = match op {
            F64Op::Dadd => "bit_cast<double>(_ds1) + bit_cast<double>(_ds2)".to_string(),
            F64Op::Dsub => "bit_cast<double>(_ds1) - bit_cast<double>(_ds2)".to_string(),
            F64Op::Dmul => "bit_cast<double>(_ds1) * bit_cast<double>(_ds2)".to_string(),
            F64Op::Dma => {
                let s3_lo = reg(rs3.unwrap_or(0));
                let s3_hi = reg(rs3.unwrap_or(0) + 1);
                self.line(&format!(
                    "uint64_t _ds3 = ((uint64_t){s3_hi} << 32) | (uint64_t){s3_lo};"
                ));
                "sycl::fma(bit_cast<double>(_ds1), bit_cast<double>(_ds2), bit_cast<double>(_ds3))"
                    .to_string()
            }
        };
        self.line(&format!(
            "uint64_t _dr = bit_cast<uint64_t>({result_expr});"
        ));
        self.line(&format!("{r_d_lo} = (uint32_t)_dr;"));
        self.line(&format!("{r_d_hi} = (uint32_t)(_dr >> 32);"));
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
        self.line(&format!(
            "uint64_t _ds1 = ((uint64_t){s1_hi} << 32) | (uint64_t){s1_lo};"
        ));
        match op {
            F64DivSqrtOp::Ddiv => {
                let s2 = rs2.unwrap_or(0);
                let s2_lo = reg(s2);
                let s2_hi = reg(s2 + 1);
                self.line(&format!(
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
        self.line(&format!("{r_d_lo} = (uint32_t)_dr;"));
        self.line(&format!("{r_d_hi} = (uint32_t)(_dr >> 32);"));
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
        let r_s1 = reg(rs1);
        match op {
            BitOpType::Bitcount => {
                self.line(&format!("{r_d} = sycl::popcount({r_s1});"));
            }
            BitOpType::Bitfind => {
                self.line(&format!(
                    "{r_d} = ({r_s1} == 0u) ? 0xFFFFFFFFu : sycl::ctz({r_s1});"
                ));
            }
            BitOpType::Bitrev => {
                self.line("{");
                self.indent_level += 1;
                self.line(&format!("uint32_t _v = {r_s1};"));
                self.line("_v = ((_v >> 1) & 0x55555555u) | ((_v & 0x55555555u) << 1);");
                self.line("_v = ((_v >> 2) & 0x33333333u) | ((_v & 0x33333333u) << 2);");
                self.line("_v = ((_v >> 4) & 0x0F0F0F0Fu) | ((_v & 0x0F0F0F0Fu) << 4);");
                self.line("_v = ((_v >> 8) & 0x00FF00FFu) | ((_v & 0x00FF00FFu) << 8);");
                self.line(&format!("{r_d} = (_v >> 16) | (_v << 16);"));
                self.indent_level = self.indent_level.saturating_sub(1);
                self.line("}");
            }
            BitOpType::Bfe => {
                let r_s2 = reg(rs2.unwrap_or(0));
                let r_s3 = reg(rs3.unwrap_or(0));
                self.line(&format!(
                    "{r_d} = ({r_s1} >> {r_s2}) & ((1u << {r_s3}) - 1u);"
                ));
            }
            BitOpType::Bfi => {
                let r_s2 = reg(rs2.unwrap_or(0));
                let r_s3 = reg(rs3.unwrap_or(0));
                let r_s4 = reg(rs4.unwrap_or(0));
                self.line(&format!(
                    "{r_d} = ({r_s2} & ~(((1u << {r_s4}) - 1u) << {r_s3})) | (({r_s1} & ((1u << {r_s4}) - 1u)) << {r_s3});"
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
            CmpOp::Ord => format!("{p_d} = !sycl::isnan({f1}) && !sycl::isnan({f2});"),
            CmpOp::Unord => format!("{p_d} = sycl::isnan({f1}) || sycl::isnan({f2});"),
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
            CvtType::F32F16 => format!("{r_d} = rhi((half)rf({r_s1}));"),
            CvtType::F16F32 => format!("{r_d} = ri((float)rh({r_s1}));"),
            CvtType::F32Bf16 => {
                format!("{r_d} = ri(bit_cast<float>(({r_s1} & 0xFFFFu) << 16));")
            }
            CvtType::Bf16F32 => {
                format!("{r_d} = (bit_cast<uint32_t>(rf({r_s1})) >> 16);")
            }
            CvtType::F32F64 => {
                let r_d1 = reg(rd + 1);
                self.line("{");
                self.indent_level += 1;
                self.line(&format!(
                    "uint64_t _dr = bit_cast<uint64_t>((double)rf({r_s1}));"
                ));
                self.line(&format!("{r_d} = (uint32_t)_dr;"));
                self.line(&format!("{r_d1} = (uint32_t)(_dr >> 32);"));
                self.indent_level = self.indent_level.saturating_sub(1);
                self.line("}");
                return;
            }
            CvtType::F64F32 => {
                let s1_hi = reg(rs1 + 1);
                self.line("{");
                self.indent_level += 1;
                self.line(&format!(
                    "uint64_t _ds = ((uint64_t){s1_hi} << 32) | (uint64_t){r_s1};"
                ));
                self.line(&format!("{r_d} = ri((float)bit_cast<double>(_ds));"));
                self.indent_level = self.indent_level.saturating_sub(1);
                self.line("}");
                return;
            }
        };
        self.line(&expr);
    }
}
