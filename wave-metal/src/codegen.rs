// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Translate decoded WAVE operations into Metal.

use std::fmt::Write;

use wave_decode::opcodes::{
    Bf16Op, Bf16PackedOp, BitOpType, CmpOp, CvtType, F16Op, F16PackedOp, F64DivSqrtOp, F64Op,
    FUnaryOp, Scope, WaveOpType,
};
use wave_decode::{DecodedInstruction, KernelInfo, Operation};

use crate::control_flow;
use crate::intrinsics;
use crate::memory;
use crate::registers::{self, pred, reg};
use crate::CompileError;

#[derive(Debug, Clone, Copy)]
enum PlanStep {
    Local { id: usize },
    Shuffle { id: usize },
    Enter { condition: u16 },
    Switch,
    Leave,
}

#[derive(Debug)]
struct MaskPlan {
    core_start: usize,
    core_end: usize,
    steps: Vec<PlanStep>,
}

fn instruction_guard(instruction: &DecodedInstruction) -> Option<u16> {
    (instruction.predicate != 0 || instruction.predicate_negated)
        .then_some(2 * u16::from(instruction.predicate) + u16::from(instruction.predicate_negated))
}

fn is_total_local(operation: &Operation) -> bool {
    matches!(
        operation,
        Operation::Iadd { .. }
            | Operation::Isub { .. }
            | Operation::Imul { .. }
            | Operation::ImulHi { .. }
            | Operation::Imad { .. }
            | Operation::And { .. }
            | Operation::Or { .. }
            | Operation::Xor { .. }
            | Operation::Not { .. }
            | Operation::Shl { .. }
            | Operation::Shr { .. }
            | Operation::Select { .. }
            | Operation::Mov { .. }
            | Operation::MovImm { .. }
            | Operation::Nop
    ) || matches!(
        operation,
        Operation::MovSr {
            sr_index: 0..=15,
            ..
        }
    ) || matches!(
        operation,
        Operation::Icmp {
            op: CmpOp::Eq | CmpOp::Ne | CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge,
            ..
        } | Operation::Ucmp {
            op: CmpOp::Eq | CmpOp::Ne | CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge,
            ..
        }
    )
}

fn is_core_instruction(instruction: &DecodedInstruction) -> bool {
    is_total_local(&instruction.operation)
        || matches!(
            instruction.operation,
            Operation::WaveOp {
                op: WaveOpType::Shuffle,
                rs2: Some(_),
                ..
            }
        )
        || (instruction_guard(instruction).is_none()
            && matches!(
                instruction.operation,
                Operation::If { .. } | Operation::Else | Operation::Endif
            ))
}

impl MaskPlan {
    fn build(instructions: &[DecodedInstruction]) -> Result<Option<Self>, CompileError> {
        let terminal = instructions.last().is_some_and(|instruction| {
            instruction_guard(instruction).is_none()
                && matches!(instruction.operation, Operation::Halt | Operation::Return)
        });
        let body_end = instructions.len() - usize::from(terminal);
        let body = &instructions[..body_end];
        let core_start = body
            .iter()
            .rposition(|instruction| matches!(instruction.operation, Operation::DeviceLoad { .. }))
            .map_or(0, |index| index + 1);
        let core_end = body
            .iter()
            .position(|instruction| matches!(instruction.operation, Operation::DeviceStore { .. }))
            .unwrap_or(body_end);
        if core_start > core_end {
            return Ok(None);
        }
        let prologue_valid = body[..core_start].iter().all(|instruction| {
            instruction_guard(instruction).is_none()
                && (is_total_local(&instruction.operation)
                    || matches!(instruction.operation, Operation::DeviceLoad { .. }))
        });
        let epilogue_valid = body[core_end..].iter().all(|instruction| {
            instruction_guard(instruction).is_none()
                && (is_total_local(&instruction.operation)
                    || matches!(instruction.operation, Operation::DeviceStore { .. }))
        });
        if !prologue_valid
            || !epilogue_valid
            || !body[core_start..core_end].iter().all(is_core_instruction)
        {
            return Ok(None);
        }
        let mut steps = Vec::new();
        let mut branches = Vec::new();
        for (id, instruction) in instructions
            .iter()
            .enumerate()
            .take(core_end)
            .skip(core_start)
        {
            match instruction.operation {
                Operation::If { ps } => {
                    let condition = 2 * u16::from(ps);
                    steps.push(PlanStep::Enter { condition });
                    branches.push(false);
                }
                Operation::Else => {
                    let Some(seen_else) = branches.last_mut() else {
                        return Err(Self::control_error(id, "else has no matching if"));
                    };
                    if *seen_else {
                        return Err(Self::control_error(id, "if has more than one else"));
                    }
                    *seen_else = true;
                    steps.push(PlanStep::Switch);
                }
                Operation::Endif => {
                    let Some(seen_else) = branches.pop() else {
                        return Err(Self::control_error(id, "endif has no matching if"));
                    };
                    if !seen_else {
                        steps.push(PlanStep::Switch);
                    }
                    steps.push(PlanStep::Leave);
                }
                _ => Self::push_leaf(id, instruction, &mut steps),
            }
        }
        if !branches.is_empty() {
            return Err(Self::control_error(core_end, "if has no matching endif"));
        }
        Ok(Some(Self {
            core_start,
            core_end,
            steps,
        }))
    }

    fn push_leaf(id: usize, instruction: &DecodedInstruction, steps: &mut Vec<PlanStep>) {
        let guard = instruction_guard(instruction);
        let shuffle = matches!(
            instruction.operation,
            Operation::WaveOp {
                op: WaveOpType::Shuffle,
                ..
            }
        );
        if let Some(condition) = guard {
            steps.push(PlanStep::Enter { condition });
        }
        steps.push(if shuffle {
            PlanStep::Shuffle { id }
        } else {
            PlanStep::Local { id }
        });
        if guard.is_some() {
            steps.extend([PlanStep::Switch, PlanStep::Leave]);
        }
    }

    fn control_error(index: usize, detail: &str) -> CompileError {
        CompileError::UnsupportedOperation(format!("instruction {index}: {detail}"))
    }
}

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
        self.generate_with_initial_registers(instructions, kernel, &[])
    }

    /// Generate a kernel with initial argument-register values.
    ///
    /// # Errors
    ///
    /// Returns an error if the argument list exceeds the register address space
    /// or the instruction stream contains an unsupported operation.
    pub fn generate_with_initial_registers(
        &mut self,
        instructions: &[DecodedInstruction],
        kernel: &KernelInfo,
        initial_registers: &[u32],
    ) -> Result<String, CompileError> {
        if initial_registers.len() > 256 {
            return Err(CompileError::UnsupportedOperation(
                "initial arguments exceed the register address space".into(),
            ));
        }
        self.output.clear();
        self.indent_level = 1;
        let mask_plan = MaskPlan::build(instructions)?;

        writeln!(
            self.output,
            "{}",
            crate::kernel::emit_kernel_signature(&kernel.name)
        )
        .expect("writing to String cannot fail");

        self.output.push_str(&registers::emit_declarations(
            kernel.register_count,
            kernel.local_memory_size,
        ));
        for (value, index) in initial_registers.iter().zip(0..kernel.register_count) {
            self.line_args(format_args!("r{index} = {value}u;"));
        }
        self.line("");

        if let Some(plan) = &mask_plan {
            for instruction in &instructions[..plan.core_start] {
                self.emit_instruction(instruction)?;
            }
            self.emit_mask_plan(plan, instructions)?;
            for instruction in &instructions[plan.core_end..] {
                self.emit_instruction(instruction)?;
            }
        } else {
            for instruction in instructions {
                self.emit_instruction(instruction)?;
            }
        }

        writeln!(self.output, "{}", crate::kernel::emit_kernel_footer())
            .expect("writing to String cannot fail");

        Ok(std::mem::take(&mut self.output))
    }

    fn emit_mask_plan(
        &mut self,
        plan: &MaskPlan,
        instructions: &[DecodedInstruction],
    ) -> Result<(), CompileError> {
        self.line("bool _wave_active = true;");
        let mut frames = Vec::new();
        let mut next_frame = 0;
        for step in &plan.steps {
            match *step {
                PlanStep::Local { id } => {
                    self.line("if (_wave_active) {");
                    self.indent();
                    self.emit_operation(&instructions[id].operation)?;
                    self.dedent();
                    self.line("}");
                }
                PlanStep::Shuffle { id } => {
                    let Operation::WaveOp {
                        rd,
                        rs1,
                        rs2: Some(rs2),
                        ..
                    } = instructions[id].operation
                    else {
                        unreachable!("mask plans contain only indexed shuffle operations");
                    };
                    self.line_args(format_args!(
                        "const uint32_t _wave_source_{id} = {};",
                        reg(rs1)
                    ));
                    self.line_args(format_args!(
                        "const uint _wave_index_{id} = select(lane_id, {}, _wave_active);",
                        reg(rs2)
                    ));
                    self.line_args(format_args!(
                        "const uint32_t _wave_gathered_{id} = simd_shuffle(_wave_source_{id}, _wave_index_{id});"
                    ));
                    self.line_args(format_args!(
                        "{} = select({}, _wave_gathered_{id}, _wave_active);",
                        reg(rd),
                        reg(rd)
                    ));
                }
                PlanStep::Enter { condition } => {
                    let frame = next_frame;
                    next_frame += 1;
                    let predicate = format!("p{}", condition / 2);
                    let condition = if condition % 2 == 0 {
                        predicate
                    } else {
                        format!("!{predicate}")
                    };
                    self.line_args(format_args!(
                        "const bool _wave_condition_{frame} = {condition};"
                    ));
                    self.line_args(format_args!(
                        "const bool _wave_entry_{frame} = _wave_active;"
                    ));
                    self.line_args(format_args!(
                        "const bool _wave_else_{frame} = _wave_active && !_wave_condition_{frame};"
                    ));
                    self.line_args(format_args!(
                        "_wave_active = _wave_active && _wave_condition_{frame};"
                    ));
                    frames.push(frame);
                }
                PlanStep::Switch => {
                    let frame = frames
                        .last()
                        .expect("mask plan construction balances every switch");
                    self.line_args(format_args!("_wave_active = _wave_else_{frame};"));
                }
                PlanStep::Leave => {
                    let frame = frames
                        .pop()
                        .expect("mask plan construction balances every leave");
                    self.line_args(format_args!("_wave_active = _wave_entry_{frame};"));
                }
            }
        }
        Ok(())
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

    fn indent(&mut self) {
        self.indent_level += 1;
    }

    fn dedent(&mut self) {
        self.indent_level = self.indent_level.saturating_sub(1);
    }

    fn emit_instruction(&mut self, instr: &DecodedInstruction) -> Result<(), CompileError> {
        if instr.predicate != 0 || instr.predicate_negated {
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
                    "{} = ri(fma(rf({}), rf({}), rf({})));",
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
                self.line_args(format_args!("{} = ri(abs(rf({})));", reg(*rd), reg(*rs1)));
            }
            Operation::Fmin { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = ri(min(rf({}), rf({})));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Fmax { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{} = ri(max(rf({}), rf({})));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Fclamp { rd, rs1, rs2, rs3 } => {
                self.line_args(format_args!(
                    "{} = ri(clamp(rf({}), rf({}), rf({})));",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2),
                    reg(*rs3)
                ));
            }
            Operation::Fsqrt { rd, rs1 } => {
                self.line_args(format_args!("{} = ri(sqrt(rf({})));", reg(*rd), reg(*rs1)));
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
                self.line_args(format_args!("    uint _addr = {};", reg(*rs1)));
                self.line_args(format_args!("    uint _stride = {};", reg(*rs2)));
                self.line("    for (uint _r = 0; _r < 4; _r++) {");
                self.line("        for (uint _c = 0; _c < 4; _c++) {");
                self.line("            _mma_a[_r * 4 + _c] = rf(*(threadgroup uint32_t*)(local_mem + _addr + _r * _stride + _c * 4));");
                self.line("        }");
                self.line("    }");
                self.line("}");
            }
            Operation::MmaLoadB { rd: _, rs1, rs2 } => {
                self.line("{");
                self.line_args(format_args!("    uint _addr = {};", reg(*rs1)));
                self.line_args(format_args!("    uint _stride = {};", reg(*rs2)));
                self.line("    for (uint _r = 0; _r < 4; _r++) {");
                self.line("        for (uint _c = 0; _c < 4; _c++) {");
                self.line("            _mma_b[_r * 4 + _c] = rf(*(threadgroup uint32_t*)(local_mem + _addr + _r * _stride + _c * 4));");
                self.line("        }");
                self.line("    }");
                self.line("}");
            }
            Operation::MmaStoreC { rd, rs1, rs2: _ } => {
                self.line("{");
                self.line_args(format_args!("    uint _addr = {};", reg(*rd)));
                self.line_args(format_args!("    uint _stride = {};", reg(*rs1)));
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
            _ => unreachable!("emit_operation routes only wave operations here"),
        }
    }

    fn emit_operation_control(&mut self, op: &Operation) -> Result<(), CompileError> {
        match op {
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
            Operation::FenceAcquire { scope }
            | Operation::FenceRelease { scope }
            | Operation::FenceAcqRel { scope } => {
                let scope = match scope {
                    Scope::Wave => "thread_scope_simdgroup",
                    Scope::Workgroup => "thread_scope_threadgroup",
                    Scope::Device => "thread_scope_device",
                    Scope::System => {
                        return Err(CompileError::UnsupportedOperation(
                            "Metal does not provide system-scope memory fences".into(),
                        ))
                    }
                };
                // Sequential consistency preserves acquire/release requirements on MSL 3.2.
                self.line_args(format_args!(
                    "atomic_thread_fence(mem_flags::mem_threadgroup | mem_flags::mem_device, memory_order_seq_cst, {scope});"
                ));
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
            FUnaryOp::Frsqrt => format!("{r_d} = ri(rsqrt(rf({source1})));"),
            FUnaryOp::Frcp => format!("{r_d} = ri(1.0f / rf({source1}));"),
            FUnaryOp::Ffloor => format!("{r_d} = ri(floor(rf({source1})));"),
            FUnaryOp::Fceil => format!("{r_d} = ri(ceil(rf({source1})));"),
            FUnaryOp::Fround => format!("{r_d} = ri(rint(rf({source1})));"),
            FUnaryOp::Ftrunc => format!("{r_d} = ri(trunc(rf({source1})));"),
            FUnaryOp::Ffract => format!("{r_d} = ri(fract(rf({source1})));"),
            FUnaryOp::Fsat => format!("{r_d} = ri(saturate(rf({source1})));"),
            FUnaryOp::Fsin => format!("{r_d} = ri(sin(rf({source1})));"),
            FUnaryOp::Fcos => format!("{r_d} = ri(cos(rf({source1})));"),
            FUnaryOp::Fexp2 => format!("{r_d} = ri(exp2(rf({source1})));"),
            FUnaryOp::Flog2 => format!("{r_d} = ri(log2(rf({source1})));"),
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
        let source1 = reg(rs1);
        let source2 = reg(rs2);

        self.line("{");
        self.indent();
        self.line_args(format_args!(
            "float b1_lo = as_type<float>(({source1} & 0xFFFFu) << 16);"
        ));
        self.line_args(format_args!(
            "float b1_hi = as_type<float>({source1} & 0xFFFF0000u);"
        ));
        self.line_args(format_args!(
            "float b2_lo = as_type<float>(({source2} & 0xFFFFu) << 16);"
        ));
        self.line_args(format_args!(
            "float b2_hi = as_type<float>({source2} & 0xFFFF0000u);"
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
                let source3 = reg(rs3.unwrap_or(0));
                self.line_args(format_args!(
                    "float b3_lo = as_type<float>(({source3} & 0xFFFFu) << 16);"
                ));
                self.line_args(format_args!(
                    "float b3_hi = as_type<float>({source3} & 0xFFFF0000u);"
                ));
                self.line("uint lo = (as_type<uint>(fma(b1_lo, b2_lo, b3_lo)) >> 16);");
                self.line("uint hi = (as_type<uint>(fma(b1_hi, b2_hi, b3_hi)) & 0xFFFF0000u);");
            }
        }

        self.line_args(format_args!("{r_d} = hi | lo;"));
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
        self.line_args(format_args!(
            "ulong d_s1 = ((ulong){s1_hi} << 32) | (ulong){s1_lo};"
        ));
        self.line_args(format_args!(
            "ulong d_s2 = ((ulong){s2_hi} << 32) | (ulong){s2_lo};"
        ));

        let result_expr = match op {
            F64Op::Dadd => "as_type<double>(d_s1) + as_type<double>(d_s2)".to_string(),
            F64Op::Dsub => "as_type<double>(d_s1) - as_type<double>(d_s2)".to_string(),
            F64Op::Dmul => "as_type<double>(d_s1) * as_type<double>(d_s2)".to_string(),
            F64Op::Dma => {
                let s3_lo = reg(rs3.unwrap_or(0));
                let s3_hi = reg(rs3.unwrap_or(0) + 1);
                self.line_args(format_args!(
                    "ulong d_s3 = ((ulong){s3_hi} << 32) | (ulong){s3_lo};"
                ));
                "fma(as_type<double>(d_s1), as_type<double>(d_s2), as_type<double>(d_s3))"
                    .to_string()
            }
        };

        self.line_args(format_args!(
            "ulong d_result = as_type<ulong>({result_expr});"
        ));
        self.line_args(format_args!("{r_d_lo} = (uint32_t)d_result;"));
        self.line_args(format_args!("{r_d_hi} = (uint32_t)(d_result >> 32);"));
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
        self.line_args(format_args!(
            "ulong d_s1 = ((ulong){s1_hi} << 32) | (ulong){s1_lo};"
        ));

        match op {
            F64DivSqrtOp::Ddiv => {
                let s2 = rs2.unwrap_or(0);
                let s2_lo = reg(s2);
                let s2_hi = reg(s2 + 1);
                self.line_args(format_args!(
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

        self.line_args(format_args!("{r_d_lo} = (uint32_t)d_result;"));
        self.line_args(format_args!("{r_d_hi} = (uint32_t)(d_result >> 32);"));
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
        let source1 = reg(rs1);

        match op {
            BitOpType::Bitcount => {
                self.line_args(format_args!("{r_d} = popcount({source1});"));
            }
            BitOpType::Bitfind => {
                self.line_args(format_args!(
                    "{r_d} = ({source1} == 0u) ? 0xFFFFFFFFu : ctz({source1});"
                ));
            }
            BitOpType::Bitrev => {
                self.line_args(format_args!("{r_d} = reverse_bits({source1});"));
            }
            BitOpType::Bfe => {
                let source2 = reg(rs2.unwrap_or(0));
                let source3 = reg(rs3.unwrap_or(0));
                self.line_args(format_args!(
                    "{r_d} = extract_bits({source1}, {source2}, {source3});"
                ));
            }
            BitOpType::Bfi => {
                let source2 = reg(rs2.unwrap_or(0));
                let source3 = reg(rs3.unwrap_or(0));
                let source4 = reg(rs4.unwrap_or(0));
                self.line_args(format_args!(
                    "{r_d} = insert_bits({source1}, {source2}, {source3}, {source4});"
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
            CvtType::F32F16 => {
                format!("{r_d} = ri((float)as_type<half>((ushort)({source1} & 0xFFFFu)));")
            }
            CvtType::F16F32 => format!("{r_d} = (uint32_t)as_type<ushort>(half(rf({source1})));"),
            CvtType::F32Bf16 => {
                format!("{r_d} = (as_type<uint>(rf({source1})) >> 16);")
            }
            CvtType::Bf16F32 => {
                format!("{r_d} = ri(as_type<float>(({source1} & 0xFFFFu) << 16));")
            }
            CvtType::F32F64 => {
                let r_d1 = reg(rd + 1);
                self.line("{");
                self.indent();
                self.line_args(format_args!(
                    "ulong d_result = as_type<ulong>((double)rf({source1}));"
                ));
                self.line_args(format_args!("{r_d} = (uint32_t)d_result;"));
                self.line_args(format_args!("{r_d1} = (uint32_t)(d_result >> 32);"));
                self.dedent();
                self.line("}");
                return;
            }
            CvtType::F64F32 => {
                let s1_hi = reg(rs1 + 1);
                self.line("{");
                self.indent();
                self.line_args(format_args!(
                    "ulong d_s1 = ((ulong){s1_hi} << 32) | (ulong){source1};"
                ));
                self.line_args(format_args!("{r_d} = ri((float)as_type<double>(d_s1));"));
                self.dedent();
                self.line("}");
                return;
            }
        };
        self.line(&expr);
    }
}
