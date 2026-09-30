// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Translate decoded WAVE operations into PTX.

use std::fmt::Write;

use wave_decode::opcodes::{
    Bf16Op, Bf16PackedOp, BitOpType, CmpOp, CvtType, F16Op, F16PackedOp, F64DivSqrtOp, F64Op,
    FUnaryOp, WaveOpType,
};
use wave_decode::{DecodedInstruction, KernelInfo, Operation};

use crate::control_flow::ControlFlowState;
use crate::intrinsics;
use crate::memory;
use crate::registers::{self, freg, pred, reg};
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
    /// call instructions, unknown opcodes or malformed control in a mask-plan core.
    /// Returns `CompileError::InvalidControlFlow` for unmatched or unterminated
    /// structured blocks in the fallback emitter.
    pub fn generate(
        &mut self,
        instructions: &[DecodedInstruction],
        kernel: &KernelInfo,
    ) -> Result<String, CompileError> {
        self.output.clear();
        self.indent_level = 1;
        self.cf = ControlFlowState::new();
        let mask_plan = MaskPlan::build(instructions)?;

        self.output
            .push_str(&crate::kernel::emit_entry_start(&kernel.name));
        if let Some(plan) = &mask_plan {
            self.emit_mask_registers(plan);
        }
        self.output.push_str(&registers::emit_declarations(
            kernel.register_count,
            kernel.local_memory_size,
        ));
        self.line("");

        if let Some(plan) = &mask_plan {
            self.initialize_wave_registers(kernel.register_count);
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

        self.cf.finish()?;
        self.output.push_str(crate::kernel::emit_entry_end());

        Ok(std::mem::take(&mut self.output))
    }

    fn emit_mask_registers(&mut self, plan: &MaskPlan) {
        self.line(".reg .pred %wave_active;");
        self.line(".reg .b32 %wave_lane, %wave_source, %wave_index, %wave_value;");
        let frame_count = plan
            .steps
            .iter()
            .filter(|step| matches!(step, PlanStep::Enter { .. }))
            .count();
        for frame in 0..frame_count {
            self.line_args(format_args!(
                ".reg .pred %wave_condition_{frame}, %wave_entry_{frame}, %wave_else_{frame};"
            ));
        }
    }

    fn initialize_wave_registers(&mut self, register_count: u32) {
        for index in 0..register_count {
            self.line_args(format_args!("mov.b32 %r{index}, 0;"));
        }
        for index in 0..4 {
            self.line_args(format_args!("setp.ne.u32 %p{index}, 0, 0;"));
        }
    }

    fn emit_mask_plan(
        &mut self,
        plan: &MaskPlan,
        instructions: &[DecodedInstruction],
    ) -> Result<(), CompileError> {
        self.line("setp.eq.u32 %wave_active, 0, 0;");
        self.line("mov.u32 %wave_lane, %laneid;");
        let mut frames = Vec::new();
        let mut next_frame = 0;
        for step in &plan.steps {
            match *step {
                PlanStep::Local { id } => {
                    self.emit_operation_with_prefix(&instructions[id].operation, "@%wave_active ")?;
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
                    self.line_args(format_args!("mov.b32 %wave_source, {};", reg(rs1)));
                    self.line_args(format_args!(
                        "selp.b32 %wave_index, {}, %wave_lane, %wave_active;",
                        reg(rs2)
                    ));
                    self.line(
                        "shfl.sync.idx.b32 %wave_value, %wave_source, %wave_index, 31, 0xFFFFFFFF;",
                    );
                    self.line_args(format_args!(
                        "@%wave_active mov.b32 {}, %wave_value;",
                        reg(rd)
                    ));
                }
                PlanStep::Enter { condition } => {
                    let frame = next_frame;
                    next_frame += 1;
                    let predicate = condition / 2;
                    if condition % 2 == 0 {
                        self.line_args(format_args!(
                            "and.pred %wave_condition_{frame}, %p{predicate}, %p{predicate};"
                        ));
                    } else {
                        self.line_args(format_args!(
                            "not.pred %wave_condition_{frame}, %p{predicate};"
                        ));
                    }
                    self.line_args(format_args!(
                        "and.pred %wave_entry_{frame}, %wave_active, %wave_active;"
                    ));
                    self.line_args(format_args!(
                        "not.pred %wave_else_{frame}, %wave_condition_{frame};"
                    ));
                    self.line_args(format_args!(
                        "and.pred %wave_else_{frame}, %wave_entry_{frame}, %wave_else_{frame};"
                    ));
                    self.line_args(format_args!(
                        "and.pred %wave_active, %wave_entry_{frame}, %wave_condition_{frame};"
                    ));
                    frames.push(frame);
                }
                PlanStep::Switch => {
                    let frame = frames
                        .last()
                        .expect("mask plan construction balances every switch");
                    self.line_args(format_args!(
                        "and.pred %wave_active, %wave_else_{frame}, %wave_else_{frame};"
                    ));
                }
                PlanStep::Leave => {
                    let frame = frames
                        .pop()
                        .expect("mask plan construction balances every leave");
                    self.line_args(format_args!(
                        "and.pred %wave_active, %wave_entry_{frame}, %wave_entry_{frame};"
                    ));
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
            if item.ends_with(':') {
                writeln!(self.output, "{item}").expect("writing to String cannot fail");
            } else {
                self.line(item);
            }
        }
    }

    fn pred_prefix(instr: &DecodedInstruction) -> String {
        if instr.predicate == 0 && !instr.predicate_negated {
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

        if is_control_flow && (instr.predicate != 0 || instr.predicate_negated) {
            return Err(CompileError::UnsupportedOperation(
                "predicated control flow instructions are not supported in PTX backend".to_string(),
            ));
        }

        self.emit_operation(instr)?;
        Ok(())
    }

    fn emit_operation(&mut self, instr: &DecodedInstruction) -> Result<(), CompileError> {
        let pp = Self::pred_prefix(instr);
        self.emit_operation_with_prefix(&instr.operation, &pp)
    }

    fn emit_operation_with_prefix(&mut self, op: &Operation, pp: &str) -> Result<(), CompileError> {
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
            | Operation::Iclamp { .. } => self.emit_operation_integer(op, pp),
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
            | Operation::FUnary { .. } => self.emit_operation_float(op, pp),
            Operation::F16 { .. }
            | Operation::F16Packed { .. }
            | Operation::Bf16 { .. }
            | Operation::Bf16Packed { .. } => self.emit_operation_half(op, pp),
            Operation::F64 { .. } | Operation::F64DivSqrt { .. } => {
                self.emit_operation_double(op, pp);
            }
            Operation::And { .. }
            | Operation::Or { .. }
            | Operation::Xor { .. }
            | Operation::Not { .. }
            | Operation::Shl { .. }
            | Operation::Shr { .. }
            | Operation::Sar { .. }
            | Operation::BitOp { .. } => self.emit_operation_bitwise(op, pp),
            Operation::Icmp { .. }
            | Operation::Ucmp { .. }
            | Operation::Fcmp { .. }
            | Operation::Select { .. }
            | Operation::Cvt { .. } => self.emit_operation_compare(op, pp),
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
            | Operation::MmaCompute { .. } => self.emit_operation_wave(op, pp),
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
            | Operation::Nop => self.emit_operation_control(op, pp)?,
            Operation::Mov { .. }
            | Operation::MovImm { .. }
            | Operation::MovSr { .. }
            | Operation::Unknown { .. } => self.emit_operation_move(op, pp)?,
        }
        Ok(())
    }

    fn emit_operation_integer(&mut self, op: &Operation, pp: &str) {
        match op {
            Operation::Iadd { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{pp}add.s32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Isub { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{pp}sub.s32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Imul { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{pp}mul.lo.s32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::ImulHi { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{pp}mul.hi.u32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Imad { rd, rs1, rs2, rs3 } => {
                self.line_args(format_args!(
                    "{pp}mad.lo.s32 {}, {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2),
                    reg(*rs3)
                ));
            }
            Operation::Idiv { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{pp}div.s32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Imod { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{pp}rem.s32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Ineg { rd, rs1 } => {
                self.line_args(format_args!("{pp}neg.s32 {}, {};", reg(*rd), reg(*rs1)));
            }
            Operation::Iabs { rd, rs1 } => {
                self.line_args(format_args!("{pp}abs.s32 {}, {};", reg(*rd), reg(*rs1)));
            }
            Operation::Imin { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{pp}min.s32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Imax { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{pp}max.s32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Iclamp { rd, rs1, rs2, rs3 } => {
                self.line_args(format_args!(
                    "{pp}max.s32 %t0, {}, {};",
                    reg(*rs1),
                    reg(*rs2)
                ));
                self.line_args(format_args!(
                    "{pp}min.s32 {}, %t0, {};",
                    reg(*rd),
                    reg(*rs3)
                ));
            }
            _ => unreachable!("emit_operation routes only integer operations here"),
        }
    }

    fn emit_operation_float(&mut self, op: &Operation, pp: &str) {
        match op {
            Operation::Fadd { rd, rs1, rs2 } => {
                // Explicit rounding prevents PTX from contracting separate WAVE operations.
                self.emit_float_binop(pp, "add.rn.f32", *rd, *rs1, *rs2);
            }
            Operation::Fsub { rd, rs1, rs2 } => {
                self.emit_float_binop(pp, "sub.rn.f32", *rd, *rs1, *rs2);
            }
            Operation::Fmul { rd, rs1, rs2 } => {
                self.emit_float_binop(pp, "mul.rn.f32", *rd, *rs1, *rs2);
            }
            Operation::Fma { rd, rs1, rs2, rs3 } => {
                self.line_args(format_args!("{pp}mov.b32 {}, {};", freg(*rs1), reg(*rs1)));
                self.line_args(format_args!("{pp}mov.b32 {}, {};", freg(*rs2), reg(*rs2)));
                self.line_args(format_args!("{pp}mov.b32 {}, {};", freg(*rs3), reg(*rs3)));
                self.line_args(format_args!(
                    "{pp}fma.rn.f32 {}, {}, {}, {};",
                    freg(*rd),
                    freg(*rs1),
                    freg(*rs2),
                    freg(*rs3)
                ));
                self.line_args(format_args!("{pp}mov.b32 {}, {};", reg(*rd), freg(*rd)));
            }
            Operation::Fdiv { rd, rs1, rs2 } => {
                self.emit_float_binop(pp, "div.approx.f32", *rd, *rs1, *rs2);
            }
            Operation::Fneg { rd, rs1 } => {
                self.emit_float_unaryop(pp, "neg.f32", *rd, *rs1);
            }
            Operation::Fabs { rd, rs1 } => {
                self.emit_float_unaryop(pp, "abs.f32", *rd, *rs1);
            }
            Operation::Fmin { rd, rs1, rs2 } => {
                self.emit_float_binop(pp, "min.f32", *rd, *rs1, *rs2);
            }
            Operation::Fmax { rd, rs1, rs2 } => {
                self.emit_float_binop(pp, "max.f32", *rd, *rs1, *rs2);
            }
            Operation::Fclamp { rd, rs1, rs2, rs3 } => {
                self.line_args(format_args!("{pp}mov.b32 {}, {};", freg(*rs1), reg(*rs1)));
                self.line_args(format_args!("{pp}mov.b32 {}, {};", freg(*rs2), reg(*rs2)));
                self.line_args(format_args!("{pp}mov.b32 {}, {};", freg(*rs3), reg(*rs3)));
                self.line_args(format_args!(
                    "{pp}max.f32 %ft0, {}, {};",
                    freg(*rs1),
                    freg(*rs2)
                ));
                self.line_args(format_args!(
                    "{pp}min.f32 {}, %ft0, {};",
                    freg(*rd),
                    freg(*rs3)
                ));
                self.line_args(format_args!("{pp}mov.b32 {}, {};", reg(*rd), freg(*rd)));
            }
            Operation::Fsqrt { rd, rs1 } => {
                self.emit_float_unaryop(pp, "sqrt.approx.f32", *rd, *rs1);
            }
            Operation::FUnary { op, rd, rs1 } => {
                self.emit_funary(pp, *op, *rd, *rs1);
            }
            _ => unreachable!("emit_operation routes only float operations here"),
        }
    }

    fn emit_operation_half(&mut self, op: &Operation, pp: &str) {
        match op {
            Operation::F16 {
                op,
                rd,
                rs1,
                rs2,
                rs3,
            } => {
                self.emit_f16(pp, *op, *rd, *rs1, *rs2, *rs3);
            }
            Operation::F16Packed {
                op,
                rd,
                rs1,
                rs2,
                rs3,
            } => {
                self.emit_f16_packed(pp, *op, *rd, *rs1, *rs2, *rs3);
            }
            Operation::Bf16 {
                op,
                rd,
                rs1,
                rs2,
                rs3,
            } => {
                self.emit_bf16(pp, *op, *rd, *rs1, *rs2, *rs3);
            }
            Operation::Bf16Packed {
                op,
                rd,
                rs1,
                rs2,
                rs3,
            } => {
                self.emit_bf16_packed(pp, *op, *rd, *rs1, *rs2, *rs3);
            }
            _ => unreachable!("emit_operation routes only half operations here"),
        }
    }

    fn emit_operation_double(&mut self, op: &Operation, pp: &str) {
        match op {
            Operation::F64 {
                op,
                rd,
                rs1,
                rs2,
                rs3,
            } => {
                self.emit_f64(pp, *op, *rd, *rs1, *rs2, *rs3);
            }
            Operation::F64DivSqrt { op, rd, rs1, rs2 } => {
                self.emit_f64_div_sqrt(pp, *op, *rd, *rs1, *rs2);
            }
            _ => unreachable!("emit_operation routes only double operations here"),
        }
    }

    fn emit_operation_bitwise(&mut self, op: &Operation, pp: &str) {
        match op {
            Operation::And { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{pp}and.b32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Or { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{pp}or.b32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Xor { rd, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{pp}xor.b32 {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2)
                ));
            }
            Operation::Not { rd, rs1 } => {
                self.line_args(format_args!("{pp}not.b32 {}, {};", reg(*rd), reg(*rs1)));
            }
            Operation::Shl { rd, rs1, rs2 } => {
                self.line_args(format_args!("{pp}and.b32 %t0, {}, 31;", reg(*rs2)));
                self.line_args(format_args!(
                    "{pp}shl.b32 {}, {}, %t0;",
                    reg(*rd),
                    reg(*rs1)
                ));
            }
            Operation::Shr { rd, rs1, rs2 } => {
                self.line_args(format_args!("{pp}and.b32 %t0, {}, 31;", reg(*rs2)));
                self.line_args(format_args!(
                    "{pp}shr.u32 {}, {}, %t0;",
                    reg(*rd),
                    reg(*rs1)
                ));
            }
            Operation::Sar { rd, rs1, rs2 } => {
                self.line_args(format_args!("{pp}and.b32 %t0, {}, 31;", reg(*rs2)));
                self.line_args(format_args!(
                    "{pp}shr.s32 {}, {}, %t0;",
                    reg(*rd),
                    reg(*rs1)
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
                self.emit_bitop(pp, *op, *rd, *rs1, [*rs2, *rs3, *rs4]);
            }
            _ => unreachable!("emit_operation routes only bitwise operations here"),
        }
    }

    fn emit_operation_compare(&mut self, op: &Operation, pp: &str) {
        match op {
            Operation::Icmp { op, pd, rs1, rs2 } => {
                self.emit_int_cmp(pp, "s32", *op, *pd, *rs1, *rs2);
            }
            Operation::Ucmp { op, pd, rs1, rs2 } => {
                self.emit_int_cmp(pp, "u32", *op, *pd, *rs1, *rs2);
            }
            Operation::Fcmp { op, pd, rs1, rs2 } => {
                self.emit_fcmp(pp, *op, *pd, *rs1, *rs2);
            }
            Operation::Select { rd, ps, rs1, rs2 } => {
                self.line_args(format_args!(
                    "{pp}selp.b32 {}, {}, {}, {};",
                    reg(*rd),
                    reg(*rs1),
                    reg(*rs2),
                    pred(*ps)
                ));
            }
            Operation::Cvt { cvt_type, rd, rs1 } => {
                self.emit_cvt(pp, *cvt_type, *rd, *rs1);
            }
            _ => unreachable!("emit_operation routes only compare operations here"),
        }
    }

    fn emit_operation_load_store(&mut self, op: &Operation) {
        match op {
            Operation::LocalLoad { width, rd, addr } => {
                let stmts = memory::emit_shared_load(*width, *rd, *addr);
                self.lines(&stmts);
            }
            Operation::LocalStore { width, addr, value } => {
                let stmts = memory::emit_shared_store(*width, *addr, *value);
                self.lines(&stmts);
            }
            Operation::DeviceLoad { width, rd, addr } => {
                let stmts = memory::emit_global_load(*width, *rd, *addr);
                self.lines(&stmts);
            }
            Operation::DeviceStore { width, addr, value } => {
                let stmts = memory::emit_global_store(*width, *addr, *value);
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
            _ => unreachable!("emit_operation routes only atomic operations here"),
        }
    }

    fn emit_operation_wave(&mut self, op: &Operation, pp: &str) {
        match op {
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
            Operation::MmaLoadA { rd: _, rs1, rs2 } => {
                self.line_args(format_args!("{pp}mov.u32 %t0, _shared_mem;"));
                self.line_args(format_args!("{pp}add.u32 %t0, %t0, {};", reg(*rs1)));
                self.line_args(format_args!(
                    "{pp}wmma.load.a.sync.aligned.row.m16n16k16.f32 \
                     {{%mA0, %mA1, %mA2, %mA3, %mA4, %mA5, %mA6, %mA7}}, \
                     [%t0], {};",
                    reg(*rs2)
                ));
            }
            Operation::MmaLoadB { rd: _, rs1, rs2 } => {
                self.line_args(format_args!("{pp}mov.u32 %t0, _shared_mem;"));
                self.line_args(format_args!("{pp}add.u32 %t0, %t0, {};", reg(*rs1)));
                self.line_args(format_args!(
                    "{pp}wmma.load.b.sync.aligned.col.m16n16k16.f32 \
                     {{%mB0, %mB1, %mB2, %mB3, %mB4, %mB5, %mB6, %mB7}}, \
                     [%t0], {};",
                    reg(*rs2)
                ));
            }
            Operation::MmaStoreC { rd, rs1, rs2: _ } => {
                self.line_args(format_args!("{pp}mov.u32 %t0, _shared_mem;"));
                self.line_args(format_args!("{pp}add.u32 %t0, %t0, {};", reg(*rd)));
                self.line_args(format_args!(
                    "{pp}wmma.store.d.sync.aligned.row.m16n16k16.f32 \
                     [%t0], \
                     {{%mC0, %mC1, %mC2, %mC3, %mC4, %mC5, %mC6, %mC7}}, {};",
                    reg(*rs1)
                ));
            }
            Operation::MmaCompute {
                rd: _,
                rs1: _,
                rs2: _,
            } => {
                self.line_args(format_args!(
                    "{pp}wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32.f32.f32 \
                     {{%mC0, %mC1, %mC2, %mC3, %mC4, %mC5, %mC6, %mC7}}, \
                     {{%mA0, %mA1, %mA2, %mA3, %mA4, %mA5, %mA6, %mA7}}, \
                     {{%mB0, %mB1, %mB2, %mB3, %mB4, %mB5, %mB6, %mB7}}, \
                     {{%mC0, %mC1, %mC2, %mC3, %mC4, %mC5, %mC6, %mC7}};"
                ));
            }
            _ => unreachable!("emit_operation routes only wave operations here"),
        }
    }

    fn emit_operation_control(&mut self, op: &Operation, pp: &str) -> Result<(), CompileError> {
        match op {
            Operation::If { ps } => {
                let stmts = self.cf.emit_if(*ps);
                self.lines(&stmts);
            }
            Operation::Else => {
                let stmts = self.cf.emit_else()?;
                self.lines(&stmts);
            }
            Operation::Endif => {
                let stmts = self.cf.emit_endif()?;
                self.lines(&stmts);
            }
            Operation::Loop => {
                let stmts = self.cf.emit_loop();
                self.lines(&stmts);
            }
            Operation::Break { ps } => {
                let stmts = self.cf.emit_break(*ps)?;
                self.lines(&stmts);
            }
            Operation::Continue { ps } => {
                let stmts = self.cf.emit_continue(*ps)?;
                self.lines(&stmts);
            }
            Operation::Endloop => {
                let stmts = self.cf.emit_endloop()?;
                self.lines(&stmts);
            }
            Operation::Call { .. } => {
                return Err(CompileError::UnsupportedOperation(
                    "call instructions are not supported in the PTX backend".to_string(),
                ));
            }
            Operation::Return | Operation::Halt => {
                self.line_args(format_args!("{pp}ret;"));
            }
            Operation::Barrier => {
                self.line_args(format_args!("{pp}bar.sync 0;"));
            }
            Operation::FenceAcquire { scope }
            | Operation::FenceRelease { scope }
            | Operation::FenceAcqRel { scope } => {
                let fence = match scope {
                    wave_decode::Scope::Wave | wave_decode::Scope::Workgroup => "membar.cta;",
                    wave_decode::Scope::Device => "membar.gl;",
                    wave_decode::Scope::System => "membar.sys;",
                };
                self.line_args(format_args!("{pp}{fence}"));
            }
            Operation::Wait | Operation::Nop => {}
            _ => unreachable!("emit_operation routes only control operations here"),
        }
        Ok(())
    }

    fn emit_operation_move(&mut self, op: &Operation, pp: &str) -> Result<(), CompileError> {
        match op {
            Operation::Mov { rd, rs1 } => {
                self.line_args(format_args!("{pp}mov.b32 {}, {};", reg(*rd), reg(*rs1)));
            }
            Operation::MovImm { rd, imm } => {
                self.line_args(format_args!("{pp}mov.b32 {}, {};", reg(*rd), imm));
            }
            Operation::MovSr { rd, sr_index } => {
                let stmts = registers::emit_special_reg(*rd, *sr_index);
                for s in &stmts {
                    self.line_args(format_args!("{pp}{s}"));
                }
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

    fn emit_float_binop(&mut self, pp: &str, op: &str, rd: u8, rs1: u8, rs2: u8) {
        self.line_args(format_args!("{pp}mov.b32 {}, {};", freg(rs1), reg(rs1)));
        self.line_args(format_args!("{pp}mov.b32 {}, {};", freg(rs2), reg(rs2)));
        self.line_args(format_args!(
            "{pp}{op} {}, {}, {};",
            freg(rd),
            freg(rs1),
            freg(rs2)
        ));
        self.line_args(format_args!("{pp}mov.b32 {}, {};", reg(rd), freg(rd)));
    }

    fn emit_float_unaryop(&mut self, pp: &str, op: &str, rd: u8, rs1: u8) {
        self.line_args(format_args!("{pp}mov.b32 {}, {};", freg(rs1), reg(rs1)));
        self.line_args(format_args!("{pp}{op} {}, {};", freg(rd), freg(rs1)));
        self.line_args(format_args!("{pp}mov.b32 {}, {};", reg(rd), freg(rd)));
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
        self.line_args(format_args!("{pp}mov.b32 {}, {};", freg(rs1), reg(rs1)));
        match op {
            FUnaryOp::Ffract => {
                self.line_args(format_args!("{pp}cvt.rmi.f32.f32 %ft0, {};", freg(rs1)));
                self.line_args(format_args!(
                    "{pp}sub.rn.f32 {}, {}, %ft0;",
                    freg(rd),
                    freg(rs1)
                ));
            }
            FUnaryOp::Fsat => {
                self.line_args(format_args!("{pp}max.f32 %ft0, {}, 0f00000000;", freg(rs1)));
                self.line_args(format_args!("{pp}min.f32 {}, %ft0, 0f3F800000;", freg(rd)));
            }
            _ => {}
        }
        self.line_args(format_args!("{pp}mov.b32 {}, {};", reg(rd), freg(rd)));
    }

    fn emit_f16(&mut self, pp: &str, op: F16Op, rd: u8, rs1: u8, rs2: u8, rs3: Option<u8>) {
        self.line_args(format_args!("{pp}cvt.u16.u32 %t0, {};", reg(rs1)));
        self.line_args(format_args!("{pp}cvt.u16.u32 %t1, {};", reg(rs2)));

        let ptx_op = match op {
            F16Op::Hadd => "add.rn.f16",
            F16Op::Hsub => "sub.rn.f16",
            F16Op::Hmul => "mul.rn.f16",
            F16Op::Hma => {
                self.line_args(format_args!(
                    "{pp}cvt.u16.u32 %t2, {};",
                    reg(rs3.unwrap_or(0))
                ));
                self.line_args(format_args!("{pp}fma.rn.f16 %t0, %t0, %t1, %t2;"));
                self.line_args(format_args!("{pp}cvt.u32.u16 {}, %t0;", reg(rd)));
                return;
            }
        };

        self.line_args(format_args!("{pp}{ptx_op} %t0, %t0, %t1;"));
        self.line_args(format_args!("{pp}cvt.u32.u16 {}, %t0;", reg(rd)));
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
            F16PackedOp::Hadd2 => "add.rn.f16x2",
            F16PackedOp::Hmul2 => "mul.rn.f16x2",
            F16PackedOp::Hma2 => {
                self.line_args(format_args!(
                    "{pp}fma.rn.f16x2 {}, {}, {}, {};",
                    reg(rd),
                    reg(rs1),
                    reg(rs2),
                    reg(rs3.unwrap_or(0))
                ));
                return;
            }
        };

        self.line_args(format_args!(
            "{pp}{ptx_op} {}, {}, {};",
            reg(rd),
            reg(rs1),
            reg(rs2)
        ));
    }

    fn emit_bf16(&mut self, pp: &str, op: Bf16Op, rd: u8, rs1: u8, rs2: u8, rs3: Option<u8>) {
        self.line_args(format_args!("{pp}shl.b32 %t0, {}, 16;", reg(rs1)));
        self.line_args(format_args!("{pp}mov.b32 %ft0, %t0;"));
        self.line_args(format_args!("{pp}shl.b32 %t1, {}, 16;", reg(rs2)));
        self.line_args(format_args!("{pp}mov.b32 %ft1, %t1;"));

        let ptx_op = match op {
            Bf16Op::Badd => "add.rn.f32",
            Bf16Op::Bsub => "sub.rn.f32",
            Bf16Op::Bmul => "mul.rn.f32",
            Bf16Op::Bma => {
                self.line_args(format_args!(
                    "{pp}shl.b32 %t2, {}, 16;",
                    reg(rs3.unwrap_or(0))
                ));
                self.line_args(format_args!("{pp}mov.b32 %ft2, %t2;"));
                self.line_args(format_args!("{pp}fma.rn.f32 %ft0, %ft0, %ft1, %ft2;"));
                self.line_args(format_args!("{pp}mov.b32 %t0, %ft0;"));
                self.line_args(format_args!("{pp}shr.b32 {}, %t0, 16;", reg(rd)));
                return;
            }
        };

        self.line_args(format_args!("{pp}{ptx_op} %ft0, %ft0, %ft1;"));
        self.line_args(format_args!("{pp}mov.b32 %t0, %ft0;"));
        self.line_args(format_args!("{pp}shr.b32 {}, %t0, 16;", reg(rd)));
    }

    fn emit_bf16_packed(
        &mut self,
        pp: &str,
        op: Bf16PackedOp,
        rd: u8,
        rs1: u8,
        rs2: u8,
        rs3: Option<u8>,
    ) {
        self.line_args(format_args!("{pp}and.b32 %t0, {}, 0xFFFF;", reg(rs1)));
        self.line_args(format_args!("{pp}shl.b32 %t0, %t0, 16;"));
        self.line_args(format_args!("{pp}mov.b32 %ft0, %t0;"));
        self.line_args(format_args!("{pp}and.b32 %t0, {}, 0xFFFF0000;", reg(rs1)));
        self.line_args(format_args!("{pp}mov.b32 %ft1, %t0;"));
        self.line_args(format_args!("{pp}and.b32 %t1, {}, 0xFFFF;", reg(rs2)));
        self.line_args(format_args!("{pp}shl.b32 %t1, %t1, 16;"));
        self.line_args(format_args!("{pp}mov.b32 %ft2, %t1;"));
        self.line_args(format_args!("{pp}and.b32 %t1, {}, 0xFFFF0000;", reg(rs2)));
        self.line_args(format_args!("{pp}mov.b32 %ft3, %t1;"));

        match op {
            Bf16PackedOp::Badd2 => {
                self.line_args(format_args!("{pp}add.rn.f32 %ft0, %ft0, %ft2;"));
                self.line_args(format_args!("{pp}add.rn.f32 %ft1, %ft1, %ft3;"));
            }
            Bf16PackedOp::Bmul2 => {
                self.line_args(format_args!("{pp}mul.rn.f32 %ft0, %ft0, %ft2;"));
                self.line_args(format_args!("{pp}mul.rn.f32 %ft1, %ft1, %ft3;"));
            }
            Bf16PackedOp::Bma2 => {
                let s3 = rs3.unwrap_or(0);
                self.line_args(format_args!("{pp}and.b32 %t2, {}, 0xFFFF;", reg(s3)));
                self.line_args(format_args!("{pp}shl.b32 %t2, %t2, 16;"));
                self.line_args(format_args!("{pp}mov.b32 %ft4, %t2;"));
                self.line_args(format_args!("{pp}and.b32 %t2, {}, 0xFFFF0000;", reg(s3)));
                self.line_args(format_args!("{pp}mov.b32 %ft5, %t2;"));
                self.line_args(format_args!("{pp}fma.rn.f32 %ft0, %ft0, %ft2, %ft4;"));
                self.line_args(format_args!("{pp}fma.rn.f32 %ft1, %ft1, %ft3, %ft5;"));
            }
        }

        self.line_args(format_args!("{pp}mov.b32 %t0, %ft0;"));
        self.line_args(format_args!("{pp}shr.b32 %t0, %t0, 16;"));
        self.line_args(format_args!("{pp}mov.b32 %t1, %ft1;"));
        self.line_args(format_args!("{pp}and.b32 %t1, %t1, 0xFFFF0000;"));
        self.line_args(format_args!("{pp}or.b32 {}, %t0, %t1;", reg(rd)));
    }

    fn emit_f64(&mut self, pp: &str, op: F64Op, rd: u8, rs1: u8, rs2: u8, rs3: Option<u8>) {
        self.line_args(format_args!(
            "{pp}mov.b64 %rd2, {{{}, {}}};",
            reg(rs1),
            reg(rs1 + 1)
        ));
        self.line_args(format_args!(
            "{pp}mov.b64 %rd3, {{{}, {}}};",
            reg(rs2),
            reg(rs2 + 1)
        ));

        let ptx_op = match op {
            F64Op::Dadd => "add.rn.f64",
            F64Op::Dsub => "sub.rn.f64",
            F64Op::Dmul => "mul.rn.f64",
            F64Op::Dma => {
                let s3 = rs3.unwrap_or(0);
                self.line_args(format_args!(
                    "{pp}mov.b64 %rd1, {{{}, {}}};",
                    reg(s3),
                    reg(s3 + 1)
                ));
                self.line_args(format_args!("{pp}fma.rn.f64 %rd2, %rd2, %rd3, %rd1;"));
                self.line_args(format_args!(
                    "{pp}mov.b64 {{{}, {}}}, %rd2;",
                    reg(rd),
                    reg(rd + 1)
                ));
                return;
            }
        };

        self.line_args(format_args!("{pp}{ptx_op} %rd2, %rd2, %rd3;"));
        self.line_args(format_args!(
            "{pp}mov.b64 {{{}, {}}}, %rd2;",
            reg(rd),
            reg(rd + 1)
        ));
    }

    fn emit_f64_div_sqrt(&mut self, pp: &str, op: F64DivSqrtOp, rd: u8, rs1: u8, rs2: Option<u8>) {
        self.line_args(format_args!(
            "{pp}mov.b64 %rd2, {{{}, {}}};",
            reg(rs1),
            reg(rs1 + 1)
        ));

        match op {
            F64DivSqrtOp::Ddiv => {
                let s2 = rs2.unwrap_or(0);
                self.line_args(format_args!(
                    "{pp}mov.b64 %rd3, {{{}, {}}};",
                    reg(s2),
                    reg(s2 + 1)
                ));
                self.line_args(format_args!("{pp}div.rn.f64 %rd2, %rd2, %rd3;"));
            }
            F64DivSqrtOp::Dsqrt => {
                self.line_args(format_args!("{pp}sqrt.rn.f64 %rd2, %rd2;"));
            }
        }

        self.line_args(format_args!(
            "{pp}mov.b64 {{{}, {}}}, %rd2;",
            reg(rd),
            reg(rd + 1)
        ));
    }

    // The bit operation format has one destination and four source operands.

    fn emit_bitop(&mut self, pp: &str, op: BitOpType, rd: u8, rs1: u8, sources: [Option<u8>; 3]) {
        let [rs2, rs3, rs4] = sources;
        match op {
            BitOpType::Bitcount => {
                self.line_args(format_args!("{pp}popc.b32 {}, {};", reg(rd), reg(rs1)));
            }
            BitOpType::Bitfind => {
                self.line_args(format_args!("{pp}neg.s32 %t0, {};", reg(rs1)));
                self.line_args(format_args!("{pp}and.b32 %t0, {}, %t0;", reg(rs1)));
                self.line_args(format_args!("{pp}bfind.u32 {}, %t0;", reg(rd)));
            }
            BitOpType::Bitrev => {
                self.line_args(format_args!("{pp}brev.b32 {}, {};", reg(rd), reg(rs1)));
            }
            BitOpType::Bfe => {
                self.line_args(format_args!(
                    "{pp}bfe.u32 {}, {}, {}, {};",
                    reg(rd),
                    reg(rs1),
                    reg(rs2.unwrap_or(0)),
                    reg(rs3.unwrap_or(0))
                ));
            }
            BitOpType::Bfi => {
                self.line_args(format_args!(
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

        self.line_args(format_args!(
            "{pp}setp.{cmp}.{ty} {}, {}, {};",
            pred(pd),
            reg(rs1),
            reg(rs2)
        ));
    }

    fn emit_fcmp(&mut self, pp: &str, op: CmpOp, pd: u8, rs1: u8, rs2: u8) {
        self.line_args(format_args!("{pp}mov.b32 {}, {};", freg(rs1), reg(rs1)));
        self.line_args(format_args!("{pp}mov.b32 {}, {};", freg(rs2), reg(rs2)));

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

        self.line_args(format_args!(
            "{pp}setp.{cmp}.f32 {}, {}, {};",
            pred(pd),
            freg(rs1),
            freg(rs2)
        ));
    }

    fn emit_cvt(&mut self, pp: &str, cvt_type: CvtType, rd: u8, rs1: u8) {
        match cvt_type {
            CvtType::F32I32 => {
                self.line_args(format_args!("{pp}mov.b32 {}, {};", freg(rs1), reg(rs1)));
                self.line_args(format_args!(
                    "{pp}cvt.rzi.s32.f32 {}, {};",
                    reg(rd),
                    freg(rs1)
                ));
            }
            CvtType::F32U32 => {
                self.line_args(format_args!("{pp}mov.b32 {}, {};", freg(rs1), reg(rs1)));
                self.line_args(format_args!(
                    "{pp}cvt.rzi.u32.f32 {}, {};",
                    reg(rd),
                    freg(rs1)
                ));
            }
            CvtType::I32F32 => {
                self.line_args(format_args!(
                    "{pp}cvt.rn.f32.s32 {}, {};",
                    freg(rd),
                    reg(rs1)
                ));
                self.line_args(format_args!("{pp}mov.b32 {}, {};", reg(rd), freg(rd)));
            }
            CvtType::U32F32 => {
                self.line_args(format_args!(
                    "{pp}cvt.rn.f32.u32 {}, {};",
                    freg(rd),
                    reg(rs1)
                ));
                self.line_args(format_args!("{pp}mov.b32 {}, {};", reg(rd), freg(rd)));
            }
            CvtType::F32F16 => {
                self.line_args(format_args!("{pp}mov.b32 {}, {};", freg(rs1), reg(rs1)));
                self.line_args(format_args!("{pp}cvt.rn.f16.f32 %t0, {};", freg(rs1)));
                self.line_args(format_args!("{pp}cvt.u32.u16 {}, %t0;", reg(rd)));
            }
            CvtType::F16F32 => {
                self.line_args(format_args!("{pp}cvt.u16.u32 %t0, {};", reg(rs1)));
                self.line_args(format_args!("{pp}cvt.f32.f16 {}, %t0;", freg(rd)));
                self.line_args(format_args!("{pp}mov.b32 {}, {};", reg(rd), freg(rd)));
            }
            CvtType::F32Bf16 => {
                self.line_args(format_args!("{pp}shl.b32 %t0, {}, 16;", reg(rs1)));
                self.line_args(format_args!("{pp}mov.b32 {}, %t0;", freg(rd)));
                self.line_args(format_args!("{pp}mov.b32 {}, {};", reg(rd), freg(rd)));
            }
            CvtType::Bf16F32 => {
                self.line_args(format_args!("{pp}mov.b32 {}, {};", freg(rs1), reg(rs1)));
                self.line_args(format_args!("{pp}mov.b32 %t0, {};", freg(rs1)));
                self.line_args(format_args!("{pp}shr.b32 {}, %t0, 16;", reg(rd)));
            }
            CvtType::F32F64 => {
                self.line_args(format_args!("{pp}mov.b32 {}, {};", freg(rs1), reg(rs1)));
                self.line_args(format_args!("{pp}cvt.f64.f32 %rd2, {};", freg(rs1)));
                self.line_args(format_args!(
                    "{pp}mov.b64 {{{}, {}}}, %rd2;",
                    reg(rd),
                    reg(rd + 1)
                ));
            }
            CvtType::F64F32 => {
                self.line_args(format_args!(
                    "{pp}mov.b64 %rd2, {{{}, {}}};",
                    reg(rs1),
                    reg(rs1 + 1)
                ));
                self.line_args(format_args!("{pp}cvt.rn.f32.f64 {}, %rd2;", freg(rd)));
                self.line_args(format_args!("{pp}mov.b32 {}, {};", reg(rd), freg(rd)));
            }
        }
    }
}
