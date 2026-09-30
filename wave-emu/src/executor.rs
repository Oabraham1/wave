// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Execute decoded instructions across the active threads in a wave.

use crate::decoder::{
    AtomicOp, Bf16Op, Bf16PackedOp, BitOpType, CmpOp, ControlOp, CvtType, DecodedInstruction,
    Decoder, F16Op, F16PackedOp, F64DivSqrtOp, F64Op, FUnaryOp, MemWidth, MiscOp, Opcode, SyncOp,
    WaveOpType, WaveReduceType, SYNC_MODIFIER_OFFSET,
};
use crate::memory::{DeviceMemory, LocalMemory};
use crate::shuffle;
use crate::stats::{ExecutionStats, InstructionCategory, TraceWriter};
use crate::wave::Wave;
use crate::EmulatorError;
use half::f16;
use num_traits::ToPrimitive;

#[derive(Debug)]
pub struct Executor<'a> {
    decoder: Decoder<'a>,
    trace: TraceWriter,
    workgroup_id: [u32; 3],
}

impl<'a> Executor<'a> {
    #[must_use]
    pub fn new(code: &'a [u8], trace_enabled: bool, workgroup_id: [u32; 3]) -> Self {
        Self {
            decoder: Decoder::new(code),
            trace: TraceWriter::new(trace_enabled),
            workgroup_id,
        }
    }

    /// Execute one instruction for a wave.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid instructions, memory accesses, control flow, or unsupported operations.
    pub fn step(
        &mut self,
        wave: &mut Wave,
        local_memory: &mut LocalMemory,
        device_memory: &mut DeviceMemory,
        stats: &mut ExecutionStats,
    ) -> Result<StepResult, EmulatorError> {
        if wave.is_halted() {
            return Ok(StepResult::Halted);
        }

        if wave.active_mask == 0 && wave.control_flow.is_empty() {
            wave.halt();
            return Ok(StepResult::Halted);
        }

        let inst = self.decoder.decode_at(wave.pc)?;

        if self.trace.is_enabled() {
            let disasm = self.decoder.disassemble(&inst);
            self.trace
                .trace_instruction(self.workgroup_id, wave.wave_id, wave.pc, &disasm);
        }

        let result = self.execute_instruction(wave, &inst, local_memory, device_memory, stats)?;

        match result {
            ExecuteResult::Continue => {
                wave.advance_pc(inst.size);
                Ok(StepResult::Continue)
            }
            ExecuteResult::Jump(target) => {
                wave.set_pc(target);
                Ok(StepResult::Continue)
            }
            ExecuteResult::Halt => {
                wave.halt();
                Ok(StepResult::Halted)
            }
            ExecuteResult::Barrier => Ok(StepResult::Barrier),
        }
    }

    fn execute_instruction(
        &mut self,
        wave: &mut Wave,
        inst: &DecodedInstruction,
        local_memory: &mut LocalMemory,
        device_memory: &mut DeviceMemory,
        stats: &mut ExecutionStats,
    ) -> Result<ExecuteResult, EmulatorError> {
        let original_mask = wave.active_mask;
        let is_control_sync = inst.opcode == Opcode::Control && inst.is_sync_op();
        let is_halt = is_control_sync && inst.modifier == SyncOp::Halt as u8 + SYNC_MODIFIER_OFFSET;

        if inst.is_predicated() {
            let pred_mask = Self::compute_predicate_mask(wave, inst.pred_reg, inst.pred_neg);

            if is_halt {
                wave.active_mask &= !pred_mask;
                if wave.active_mask == 0 {
                    return Ok(ExecuteResult::Halt);
                }
                return Ok(ExecuteResult::Continue);
            } else if is_control_sync {
                if pred_mask == 0 {
                    return Ok(ExecuteResult::Continue);
                }
            } else {
                wave.active_mask &= pred_mask;
            }
        }

        let result = self.execute_operation(wave, inst, local_memory, device_memory, stats);

        if inst.is_predicated() && !is_control_sync {
            wave.active_mask = original_mask;
        }

        result
    }
    fn execute_operation(
        &mut self,
        wave: &mut Wave,
        inst: &DecodedInstruction,
        local_memory: &mut LocalMemory,
        device_memory: &mut DeviceMemory,
        stats: &mut ExecutionStats,
    ) -> Result<ExecuteResult, EmulatorError> {
        let category = match inst.opcode {
            Opcode::Iadd
            | Opcode::Isub
            | Opcode::Imul
            | Opcode::ImulHi
            | Opcode::Idiv
            | Opcode::Imod
            | Opcode::Ineg
            | Opcode::Iabs
            | Opcode::Imin
            | Opcode::Imax => {
                Self::execute_integer_op(wave, inst);
                InstructionCategory::Integer
            }
            Opcode::Imad | Opcode::Iclamp => {
                Self::execute_integer_extended(wave, inst);
                InstructionCategory::Integer
            }
            Opcode::And
            | Opcode::Or
            | Opcode::Xor
            | Opcode::Not
            | Opcode::Shl
            | Opcode::Shr
            | Opcode::Sar => {
                Self::execute_bitwise_op(wave, inst);
                InstructionCategory::Integer
            }
            Opcode::BitOps => {
                Self::execute_bit_ops(wave, inst);
                InstructionCategory::Integer
            }
            Opcode::Fadd
            | Opcode::Fsub
            | Opcode::Fmul
            | Opcode::Fdiv
            | Opcode::Fneg
            | Opcode::Fabs
            | Opcode::Fmin
            | Opcode::Fmax
            | Opcode::Fsqrt => {
                Self::execute_float_op(wave, inst);
                InstructionCategory::Float
            }
            Opcode::Fma | Opcode::Fclamp => {
                Self::execute_float_extended(wave, inst);
                InstructionCategory::Float
            }
            Opcode::FUnaryOps => {
                Self::execute_float_unary(wave, inst);
                InstructionCategory::Float
            }
            Opcode::F16Ops | Opcode::F16PackedOps => {
                Self::execute_f16_op(wave, inst);
                InstructionCategory::Float
            }
            Opcode::Bf16Ops | Opcode::Bf16PackedOps => {
                Self::execute_bf16_op(wave, inst);
                InstructionCategory::Float
            }
            Opcode::F64Ops | Opcode::F64DivSqrt => {
                Self::execute_f64_op(wave, inst);
                InstructionCategory::Float
            }
            Opcode::Icmp | Opcode::Ucmp | Opcode::Fcmp => {
                Self::execute_compare(wave, inst);
                InstructionCategory::Integer
            }
            Opcode::Select => {
                Self::execute_select(wave, inst);
                InstructionCategory::Integer
            }
            Opcode::Cvt => {
                Self::execute_convert(wave, inst);
                InstructionCategory::Float
            }
            Opcode::WaveOp => {
                Self::execute_wave_op(wave, inst);
                InstructionCategory::WaveOp
            }
            Opcode::Control => return self.execute_control(wave, inst, stats),
            Opcode::Misc => {
                Self::execute_misc_op(wave, inst);
                stats.record_instruction(InstructionCategory::Control);
                return Ok(ExecuteResult::Continue);
            }
            Opcode::LocalLoad
            | Opcode::LocalStore
            | Opcode::DeviceLoad
            | Opcode::DeviceStore
            | Opcode::LocalAtomic
            | Opcode::DeviceAtomic
            | Opcode::Mma => {
                return Self::execute_memory(wave, inst, local_memory, device_memory, stats)
            }
        };
        stats.record_instruction(category);
        Ok(ExecuteResult::Continue)
    }

    fn execute_memory(
        wave: &mut Wave,
        inst: &DecodedInstruction,
        local_memory: &mut LocalMemory,
        device_memory: &mut DeviceMemory,
        stats: &mut ExecutionStats,
    ) -> Result<ExecuteResult, EmulatorError> {
        match inst.opcode {
            Opcode::LocalLoad => {
                Self::execute_local_load(wave, inst, local_memory, stats)?;
                Ok(ExecuteResult::Continue)
            }
            Opcode::LocalStore => {
                Self::execute_local_store(wave, inst, local_memory, stats)?;
                Ok(ExecuteResult::Continue)
            }
            Opcode::DeviceLoad => {
                Self::execute_device_load(wave, inst, device_memory, stats)?;
                Ok(ExecuteResult::Continue)
            }
            Opcode::DeviceStore => {
                Self::execute_device_store(wave, inst, device_memory, stats)?;
                Ok(ExecuteResult::Continue)
            }
            Opcode::LocalAtomic => {
                Self::execute_local_atomic(wave, inst, local_memory, stats)?;
                Ok(ExecuteResult::Continue)
            }
            Opcode::DeviceAtomic => {
                Self::execute_device_atomic(wave, inst, device_memory, stats)?;
                Ok(ExecuteResult::Continue)
            }
            Opcode::Mma => {
                Self::execute_mma(wave, inst, local_memory)?;
                stats.record_instruction(InstructionCategory::Float);
                Ok(ExecuteResult::Continue)
            }
            _ => unreachable!("execute_operation routes only memory and matrix operations here"),
        }
    }

    fn low_half(value: u32) -> u16 {
        let [a, b, ..] = value.to_le_bytes();
        u16::from_le_bytes([a, b])
    }

    fn low_word(value: impl Into<u128>) -> u32 {
        let [a, b, c, d, ..] = value.into().to_le_bytes();
        u32::from_le_bytes([a, b, c, d])
    }

    fn compute_predicate_mask(wave: &Wave, pred_reg: u8, negated: bool) -> u64 {
        let mut mask: u64 = 0;
        for lane in 0..wave.wave_width {
            if wave.is_thread_active(lane) {
                let pred = wave.threads[lane as usize].read_predicate(pred_reg);
                let value = if negated { !pred } else { pred };
                if value {
                    mask |= 1u64 << lane;
                }
            }
        }
        mask
    }

    fn execute_integer_op(wave: &mut Wave, inst: &DecodedInstruction) {
        for lane in 0..wave.wave_width {
            if !wave.is_thread_active(lane) {
                continue;
            }

            let thread = &mut wave.threads[lane as usize];
            let rs1 = thread.read_register(inst.rs1);
            let rs2 = thread.read_register(inst.rs2);

            let result = match inst.opcode {
                Opcode::Iadd => rs1.wrapping_add(rs2),
                Opcode::Isub => rs1.wrapping_sub(rs2),
                Opcode::Imul => rs1.wrapping_mul(rs2),
                Opcode::ImulHi => {
                    let wide = i64::from(rs1).wrapping_mul(i64::from(rs2));
                    Self::low_word((wide >> 32).cast_unsigned())
                }
                Opcode::Idiv => {
                    if rs2 == 0 {
                        0
                    } else {
                        rs1.cast_signed()
                            .wrapping_div(rs2.cast_signed())
                            .cast_unsigned()
                    }
                }
                Opcode::Imod => {
                    if rs2 == 0 {
                        0
                    } else {
                        rs1.cast_signed()
                            .wrapping_rem(rs2.cast_signed())
                            .cast_unsigned()
                    }
                }
                Opcode::Ineg => rs1.cast_signed().wrapping_neg().cast_unsigned(),
                Opcode::Iabs => rs1.cast_signed().unsigned_abs(),
                Opcode::Imin => rs1.cast_signed().min(rs2.cast_signed()).cast_unsigned(),
                Opcode::Imax => rs1.cast_signed().max(rs2.cast_signed()).cast_unsigned(),
                _ => 0,
            };

            thread.write_register(inst.rd, result);
        }
    }

    fn execute_integer_extended(wave: &mut Wave, inst: &DecodedInstruction) {
        for lane in 0..wave.wave_width {
            if !wave.is_thread_active(lane) {
                continue;
            }

            let thread = &mut wave.threads[lane as usize];
            let rs1 = thread.read_register(inst.rs1);
            let rs2 = thread.read_register(inst.rs2);
            let rs3 = thread.read_register(inst.rs3);

            let result = match inst.opcode {
                Opcode::Imad => rs1.wrapping_mul(rs2).wrapping_add(rs3),
                Opcode::Iclamp => {
                    let val = rs1.cast_signed();
                    let lo = rs2.cast_signed();
                    let hi = rs3.cast_signed();
                    val.clamp(lo, hi).cast_unsigned()
                }
                _ => 0,
            };

            thread.write_register(inst.rd, result);
        }
    }

    fn execute_bitwise_op(wave: &mut Wave, inst: &DecodedInstruction) {
        for lane in 0..wave.wave_width {
            if !wave.is_thread_active(lane) {
                continue;
            }

            let thread = &mut wave.threads[lane as usize];
            let rs1 = thread.read_register(inst.rs1);
            let rs2 = thread.read_register(inst.rs2);

            let result = match inst.opcode {
                Opcode::And => rs1 & rs2,
                Opcode::Or => rs1 | rs2,
                Opcode::Xor => rs1 ^ rs2,
                Opcode::Not => !rs1,
                Opcode::Shl => rs1.wrapping_shl(rs2 & 0x1F),
                Opcode::Shr => rs1.wrapping_shr(rs2 & 0x1F),
                Opcode::Sar => rs1.cast_signed().wrapping_shr(rs2 & 0x1F).cast_unsigned(),
                _ => 0,
            };

            thread.write_register(inst.rd, result);
        }
    }

    fn execute_bit_ops(wave: &mut Wave, inst: &DecodedInstruction) {
        for lane in 0..wave.wave_width {
            if !wave.is_thread_active(lane) {
                continue;
            }

            let thread = &mut wave.threads[lane as usize];
            let rs1 = thread.read_register(inst.rs1);
            let rs2 = thread.read_register(inst.rs2);
            let rs3 = thread.read_register(inst.rs3);
            let rs4 = thread.read_register(inst.rs4);

            let result = match inst.modifier {
                m if m == BitOpType::Bitcount as u8 => rs1.count_ones(),
                m if m == BitOpType::Bitfind as u8 => {
                    if rs1 == 0 {
                        u32::MAX
                    } else {
                        rs1.leading_zeros()
                    }
                }
                m if m == BitOpType::Bitrev as u8 => rs1.reverse_bits(),
                m if m == BitOpType::Bfe as u8 => {
                    let offset = rs2 & 0x1F;
                    let width = rs3 & 0x1F;
                    if width == 0 {
                        0
                    } else {
                        (rs1 >> offset) & ((1 << width) - 1)
                    }
                }
                m if m == BitOpType::Bfi as u8 => {
                    let offset = rs3 & 0x1F;
                    let width = rs4 & 0x1F;
                    if width == 0 {
                        rs1
                    } else {
                        let mask = ((1u32 << width) - 1) << offset;
                        (rs1 & !mask) | ((rs2 << offset) & mask)
                    }
                }
                _ => 0,
            };

            thread.write_register(inst.rd, result);
        }
    }

    fn execute_float_op(wave: &mut Wave, inst: &DecodedInstruction) {
        for lane in 0..wave.wave_width {
            if !wave.is_thread_active(lane) {
                continue;
            }

            let thread = &mut wave.threads[lane as usize];
            let rs1 = f32::from_bits(thread.read_register(inst.rs1));
            let rs2 = f32::from_bits(thread.read_register(inst.rs2));

            let result = match inst.opcode {
                Opcode::Fadd => rs1 + rs2,
                Opcode::Fsub => rs1 - rs2,
                Opcode::Fmul => rs1 * rs2,
                Opcode::Fdiv => {
                    if rs2 == 0.0 {
                        f32::INFINITY
                    } else {
                        rs1 / rs2
                    }
                }
                Opcode::Fneg => -rs1,
                Opcode::Fabs => rs1.abs(),
                Opcode::Fmin => rs1.min(rs2),
                Opcode::Fmax => rs1.max(rs2),
                Opcode::Fsqrt => rs1.sqrt(),
                _ => 0.0,
            };

            thread.write_register(inst.rd, result.to_bits());
        }
    }

    fn execute_float_extended(wave: &mut Wave, inst: &DecodedInstruction) {
        for lane in 0..wave.wave_width {
            if !wave.is_thread_active(lane) {
                continue;
            }

            let thread = &mut wave.threads[lane as usize];
            let rs1 = f32::from_bits(thread.read_register(inst.rs1));
            let rs2 = f32::from_bits(thread.read_register(inst.rs2));
            let rs3 = f32::from_bits(thread.read_register(inst.rs3));

            let result = match inst.opcode {
                Opcode::Fma => rs1.mul_add(rs2, rs3),
                Opcode::Fclamp => rs1.clamp(rs2, rs3),
                _ => 0.0,
            };

            thread.write_register(inst.rd, result.to_bits());
        }
    }

    fn execute_float_unary(wave: &mut Wave, inst: &DecodedInstruction) {
        for lane in 0..wave.wave_width {
            if !wave.is_thread_active(lane) {
                continue;
            }

            let thread = &mut wave.threads[lane as usize];
            let rs1 = f32::from_bits(thread.read_register(inst.rs1));

            let result = match inst.modifier {
                m if m == FUnaryOp::Frsqrt as u8 => 1.0 / rs1.sqrt(),
                m if m == FUnaryOp::Frcp as u8 => 1.0 / rs1,
                m if m == FUnaryOp::Ffloor as u8 => rs1.floor(),
                m if m == FUnaryOp::Fceil as u8 => rs1.ceil(),
                m if m == FUnaryOp::Fround as u8 => rs1.round(),
                m if m == FUnaryOp::Ftrunc as u8 => rs1.trunc(),
                m if m == FUnaryOp::Ffract as u8 => rs1.fract(),
                m if m == FUnaryOp::Fsat as u8 => rs1.clamp(0.0, 1.0),
                m if m == FUnaryOp::Fsin as u8 => rs1.sin(),
                m if m == FUnaryOp::Fcos as u8 => rs1.cos(),
                m if m == FUnaryOp::Fexp2 as u8 => rs1.exp2(),
                m if m == FUnaryOp::Flog2 as u8 => rs1.log2(),
                _ => 0.0,
            };

            thread.write_register(inst.rd, result.to_bits());
        }
    }

    fn execute_f16_op(wave: &mut Wave, inst: &DecodedInstruction) {
        for lane in 0..wave.wave_width {
            if !wave.is_thread_active(lane) {
                continue;
            }

            let thread = &mut wave.threads[lane as usize];
            let rs1_bits = thread.read_register(inst.rs1);
            let rs2_bits = thread.read_register(inst.rs2);
            let rs3_bits = thread.read_register(inst.rs3);

            let result = if inst.opcode == Opcode::F16Ops {
                let a = f16::from_bits(Self::low_half(rs1_bits));
                let b = f16::from_bits(Self::low_half(rs2_bits));
                let c = f16::from_bits(Self::low_half(rs3_bits));

                let r = match inst.modifier {
                    m if m == F16Op::Hadd as u8 => f16::from_f32(a.to_f32() + b.to_f32()),
                    m if m == F16Op::Hsub as u8 => f16::from_f32(a.to_f32() - b.to_f32()),
                    m if m == F16Op::Hmul as u8 => f16::from_f32(a.to_f32() * b.to_f32()),
                    m if m == F16Op::Hma as u8 => {
                        f16::from_f32(a.to_f32().mul_add(b.to_f32(), c.to_f32()))
                    }
                    _ => f16::ZERO,
                };
                u32::from(r.to_bits())
            } else {
                let a_lo = f16::from_bits(Self::low_half(rs1_bits));
                let a_hi = f16::from_bits((rs1_bits >> 16) as u16);
                let b_lo = f16::from_bits(Self::low_half(rs2_bits));
                let b_hi = f16::from_bits((rs2_bits >> 16) as u16);
                let c_lo = f16::from_bits(Self::low_half(rs3_bits));
                let c_hi = f16::from_bits((rs3_bits >> 16) as u16);

                let (r_lo, r_hi) = match inst.modifier {
                    m if m == F16PackedOp::Hadd2 as u8 => (
                        f16::from_f32(a_lo.to_f32() + b_lo.to_f32()),
                        f16::from_f32(a_hi.to_f32() + b_hi.to_f32()),
                    ),
                    m if m == F16PackedOp::Hmul2 as u8 => (
                        f16::from_f32(a_lo.to_f32() * b_lo.to_f32()),
                        f16::from_f32(a_hi.to_f32() * b_hi.to_f32()),
                    ),
                    m if m == F16PackedOp::Hma2 as u8 => (
                        f16::from_f32(a_lo.to_f32().mul_add(b_lo.to_f32(), c_lo.to_f32())),
                        f16::from_f32(a_hi.to_f32().mul_add(b_hi.to_f32(), c_hi.to_f32())),
                    ),
                    _ => (f16::ZERO, f16::ZERO),
                };
                u32::from(r_lo.to_bits()) | (u32::from(r_hi.to_bits()) << 16)
            };

            thread.write_register(inst.rd, result);
        }
    }

    fn bf16_to_f32(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }

    fn f32_to_bf16(val: f32) -> u16 {
        (val.to_bits() >> 16) as u16
    }

    fn execute_bf16_op(wave: &mut Wave, inst: &DecodedInstruction) {
        for lane in 0..wave.wave_width {
            if !wave.is_thread_active(lane) {
                continue;
            }

            let thread = &mut wave.threads[lane as usize];
            let rs1_bits = thread.read_register(inst.rs1);
            let rs2_bits = thread.read_register(inst.rs2);
            let rs3_bits = thread.read_register(inst.rs3);

            let result = if inst.opcode == Opcode::Bf16Ops {
                let a = Self::bf16_to_f32(Self::low_half(rs1_bits));
                let b = Self::bf16_to_f32(Self::low_half(rs2_bits));
                let c = Self::bf16_to_f32(Self::low_half(rs3_bits));

                let r = match inst.modifier {
                    m if m == Bf16Op::Badd as u8 => Self::f32_to_bf16(a + b),
                    m if m == Bf16Op::Bsub as u8 => Self::f32_to_bf16(a - b),
                    m if m == Bf16Op::Bmul as u8 => Self::f32_to_bf16(a * b),
                    m if m == Bf16Op::Bma as u8 => Self::f32_to_bf16(a.mul_add(b, c)),
                    _ => 0,
                };
                u32::from(r)
            } else {
                let a_lo = Self::bf16_to_f32(Self::low_half(rs1_bits));
                let a_hi = Self::bf16_to_f32((rs1_bits >> 16) as u16);
                let b_lo = Self::bf16_to_f32(Self::low_half(rs2_bits));
                let b_hi = Self::bf16_to_f32((rs2_bits >> 16) as u16);
                let c_lo = Self::bf16_to_f32(Self::low_half(rs3_bits));
                let c_hi = Self::bf16_to_f32((rs3_bits >> 16) as u16);

                let (r_lo, r_hi) = match inst.modifier {
                    m if m == Bf16PackedOp::Badd2 as u8 => (
                        Self::f32_to_bf16(a_lo + b_lo),
                        Self::f32_to_bf16(a_hi + b_hi),
                    ),
                    m if m == Bf16PackedOp::Bmul2 as u8 => (
                        Self::f32_to_bf16(a_lo * b_lo),
                        Self::f32_to_bf16(a_hi * b_hi),
                    ),
                    m if m == Bf16PackedOp::Bma2 as u8 => (
                        Self::f32_to_bf16(a_lo.mul_add(b_lo, c_lo)),
                        Self::f32_to_bf16(a_hi.mul_add(b_hi, c_hi)),
                    ),
                    _ => (0, 0),
                };
                u32::from(r_lo) | (u32::from(r_hi) << 16)
            };

            thread.write_register(inst.rd, result);
        }
    }

    fn execute_f64_op(wave: &mut Wave, inst: &DecodedInstruction) {
        for lane in 0..wave.wave_width {
            if !wave.is_thread_active(lane) {
                continue;
            }

            let thread = &mut wave.threads[lane as usize];

            let rs1_lo = thread.read_register(inst.rs1);
            let rs1_hi = thread.read_register(inst.rs1 + 1);
            let a = f64::from_bits((u64::from(rs1_hi) << 32) | u64::from(rs1_lo));

            let rs2_lo = thread.read_register(inst.rs2);
            let rs2_hi = thread.read_register(inst.rs2 + 1);
            let b = f64::from_bits((u64::from(rs2_hi) << 32) | u64::from(rs2_lo));

            let result = if inst.opcode == Opcode::F64Ops {
                let rs3_lo = thread.read_register(inst.rs3);
                let rs3_hi = thread.read_register(inst.rs3 + 1);
                let c = f64::from_bits((u64::from(rs3_hi) << 32) | u64::from(rs3_lo));

                match inst.modifier {
                    m if m == F64Op::Dadd as u8 => a + b,
                    m if m == F64Op::Dsub as u8 => a - b,
                    m if m == F64Op::Dmul as u8 => a * b,
                    m if m == F64Op::Dma as u8 => a.mul_add(b, c),
                    _ => 0.0,
                }
            } else {
                match inst.modifier {
                    m if m == F64DivSqrtOp::Ddiv as u8 => a / b,
                    m if m == F64DivSqrtOp::Dsqrt as u8 => a.sqrt(),
                    _ => 0.0,
                }
            };

            let bits = result.to_bits();
            thread.write_register(inst.rd, Self::low_word(bits));
            thread.write_register(inst.rd + 1, (bits >> 32) as u32);
        }
    }

    // WAVE floating comparison instructions require exact IEEE equality.
    fn execute_compare(wave: &mut Wave, inst: &DecodedInstruction) {
        for lane in 0..wave.wave_width {
            if !wave.is_thread_active(lane) {
                continue;
            }

            let thread = &mut wave.threads[lane as usize];
            let rs1 = thread.read_register(inst.rs1);
            let rs2 = thread.read_register(inst.rs2);

            let result = match inst.opcode {
                Opcode::Icmp => {
                    let a = rs1.cast_signed();
                    let b = rs2.cast_signed();
                    match inst.modifier {
                        m if m == CmpOp::Eq as u8 => a == b,
                        m if m == CmpOp::Ne as u8 => a != b,
                        m if m == CmpOp::Lt as u8 => a < b,
                        m if m == CmpOp::Le as u8 => a <= b,
                        m if m == CmpOp::Gt as u8 => a > b,
                        m if m == CmpOp::Ge as u8 => a >= b,
                        _ => false,
                    }
                }
                Opcode::Ucmp => match inst.modifier {
                    m if m == CmpOp::Lt as u8 => rs1 < rs2,
                    m if m == CmpOp::Le as u8 => rs1 <= rs2,
                    _ => false,
                },
                Opcode::Fcmp => {
                    let a = f32::from_bits(rs1);
                    let b = f32::from_bits(rs2);
                    match inst.modifier {
                        m if m == CmpOp::Eq as u8 => {
                            a.partial_cmp(&b) == Some(std::cmp::Ordering::Equal)
                        }
                        m if m == CmpOp::Ne as u8 => {
                            a.partial_cmp(&b) != Some(std::cmp::Ordering::Equal)
                        }
                        m if m == CmpOp::Lt as u8 => a < b,
                        m if m == CmpOp::Le as u8 => a <= b,
                        m if m == CmpOp::Gt as u8 => a > b,
                        m if m == CmpOp::Ord as u8 => !a.is_nan() && !b.is_nan(),
                        m if m == CmpOp::Unord as u8 => a.is_nan() || b.is_nan(),
                        _ => false,
                    }
                }
                _ => false,
            };

            thread.write_predicate(inst.rd, result);
        }
    }

    fn execute_select(wave: &mut Wave, inst: &DecodedInstruction) {
        for lane in 0..wave.wave_width {
            if !wave.is_thread_active(lane) {
                continue;
            }

            let thread = &mut wave.threads[lane as usize];
            let pred = thread.read_predicate(inst.modifier);
            let rs1 = thread.read_register(inst.rs1);
            let rs2 = thread.read_register(inst.rs2);

            let result = if pred { rs1 } else { rs2 };
            thread.write_register(inst.rd, result);
        }
    }

    // WAVE conversion instructions narrow, round, and saturate numeric values.
    fn saturating_i32(value: f32) -> i32 {
        value.to_i32().unwrap_or_else(|| {
            if value.is_nan() {
                0
            } else if value.is_sign_negative() {
                i32::MIN
            } else {
                i32::MAX
            }
        })
    }

    fn saturating_u32(value: f32) -> u32 {
        value.to_u32().unwrap_or_else(|| {
            if value.is_nan() || value.is_sign_negative() {
                0
            } else {
                u32::MAX
            }
        })
    }

    fn execute_convert(wave: &mut Wave, inst: &DecodedInstruction) {
        for lane in 0..wave.wave_width {
            if !wave.is_thread_active(lane) {
                continue;
            }

            let thread = &mut wave.threads[lane as usize];
            let rs1 = thread.read_register(inst.rs1);

            let result = match inst.modifier {
                m if m == CvtType::F32I32 as u8 => rs1
                    .cast_signed()
                    .to_f32()
                    .expect("i32 values fit the f32 exponent range")
                    .to_bits(),
                m if m == CvtType::F32U32 as u8 => rs1
                    .to_f32()
                    .expect("u32 values fit the f32 exponent range")
                    .to_bits(),
                m if m == CvtType::I32F32 as u8 => {
                    Self::saturating_i32(f32::from_bits(rs1)).cast_unsigned()
                }
                m if m == CvtType::U32F32 as u8 => Self::saturating_u32(f32::from_bits(rs1)),
                m if m == CvtType::F32F16 as u8 => {
                    f16::from_bits(Self::low_half(rs1)).to_f32().to_bits()
                }
                m if m == CvtType::F16F32 as u8 => {
                    u32::from(f16::from_f32(f32::from_bits(rs1)).to_bits())
                }
                m if m == CvtType::F32F64 as u8 => {
                    let rs1_hi = thread.read_register(inst.rs1 + 1);
                    let d = f64::from_bits((u64::from(rs1_hi) << 32) | u64::from(rs1));
                    d.to_f32()
                        .expect("f64 to f32 conversion supports every float value")
                        .to_bits()
                }
                m if m == CvtType::F64F32 as u8 => {
                    let f = f32::from_bits(rs1);
                    let d = f64::from(f);
                    let bits = d.to_bits();
                    thread.write_register(inst.rd + 1, (bits >> 32) as u32);
                    Self::low_word(bits)
                }
                m if m == CvtType::F32Bf16 as u8 => {
                    Self::bf16_to_f32(Self::low_half(rs1)).to_bits()
                }
                m if m == CvtType::Bf16F32 as u8 => {
                    u32::from(Self::f32_to_bf16(f32::from_bits(rs1)))
                }
                _ => 0,
            };

            thread.write_register(inst.rd, result);
        }
    }

    fn execute_local_load(
        wave: &mut Wave,
        inst: &DecodedInstruction,
        local_memory: &mut LocalMemory,
        stats: &mut ExecutionStats,
    ) -> Result<(), EmulatorError> {
        let width: u32 = match inst.modifier {
            m if m == MemWidth::U8 as u8 => 1,
            m if m == MemWidth::U16 as u8 => 2,
            m if m == MemWidth::U32 as u8 => 4,
            m if m == MemWidth::U64 as u8 => 8,
            _ => 4,
        };

        for lane in 0..wave.wave_width {
            if !wave.is_thread_active(lane) {
                continue;
            }

            let thread = &mut wave.threads[lane as usize];
            let addr = thread.read_register(inst.rs1);

            let value = match width {
                1 => u32::from(local_memory.read_u8(addr)?),
                2 => u32::from(local_memory.read_u16(addr)?),
                4 => local_memory.read_u32(addr)?,
                8 => {
                    let val = local_memory.read_u64(addr)?;
                    thread.write_register(inst.rd + 1, Self::low_word(val >> 32));
                    Self::low_word(val)
                }
                _ => 0,
            };

            thread.write_register(inst.rd, value);
            stats.record_local_load(u64::from(width));
        }

        stats.record_instruction(InstructionCategory::Memory);
        Ok(())
    }

    fn execute_local_store(
        wave: &mut Wave,
        inst: &DecodedInstruction,
        local_memory: &mut LocalMemory,
        stats: &mut ExecutionStats,
    ) -> Result<(), EmulatorError> {
        let width: u32 = match inst.modifier {
            m if m == MemWidth::U8 as u8 => 1,
            m if m == MemWidth::U16 as u8 => 2,
            m if m == MemWidth::U32 as u8 => 4,
            m if m == MemWidth::U64 as u8 => 8,
            _ => 4,
        };

        for lane in 0..wave.wave_width {
            if !wave.is_thread_active(lane) {
                continue;
            }

            let thread = &wave.threads[lane as usize];
            let addr = thread.read_register(inst.rs1);
            let value = thread.read_register(inst.rs2);

            match width {
                1 => local_memory.write_u8(addr, value.to_le_bytes()[0])?,
                2 => local_memory.write_u16(addr, Self::low_half(value))?,
                4 => local_memory.write_u32(addr, value)?,
                8 => {
                    let hi = thread.read_register(inst.rs2 + 1);
                    let val = (u64::from(hi) << 32) | u64::from(value);
                    local_memory.write_u64(addr, val)?;
                }
                _ => {}
            }

            stats.record_local_store(u64::from(width));
        }

        stats.record_instruction(InstructionCategory::Memory);
        Ok(())
    }

    fn execute_device_load(
        wave: &mut Wave,
        inst: &DecodedInstruction,
        device_memory: &mut DeviceMemory,
        stats: &mut ExecutionStats,
    ) -> Result<(), EmulatorError> {
        let width: u32 = match inst.modifier {
            m if m == MemWidth::U8 as u8 => 1,
            m if m == MemWidth::U16 as u8 => 2,
            m if m == MemWidth::U32 as u8 => 4,
            m if m == MemWidth::U64 as u8 => 8,
            m if m == MemWidth::U128 as u8 => 16,
            _ => 4,
        };

        for lane in 0..wave.wave_width {
            if !wave.is_thread_active(lane) {
                continue;
            }

            let thread = &mut wave.threads[lane as usize];
            let addr = u64::from(thread.read_register(inst.rs1));

            match width {
                1 => {
                    let val = device_memory.read_u8(addr)?;
                    thread.write_register(inst.rd, u32::from(val));
                }
                2 => {
                    let val = device_memory.read_u16(addr)?;
                    thread.write_register(inst.rd, u32::from(val));
                }
                4 => {
                    let val = device_memory.read_u32(addr)?;
                    thread.write_register(inst.rd, val);
                }
                8 => {
                    let val = device_memory.read_u64(addr)?;
                    thread.write_register(inst.rd, Self::low_word(val));
                    thread.write_register(inst.rd + 1, Self::low_word(val >> 32));
                }
                16 => {
                    let val = device_memory.read_u128(addr)?;
                    thread.write_register(inst.rd, Self::low_word(val));
                    thread.write_register(inst.rd + 1, Self::low_word(val >> 32));
                    thread.write_register(inst.rd + 2, Self::low_word(val >> 64));
                    thread.write_register(inst.rd + 3, (val >> 96) as u32);
                }
                _ => {}
            }

            stats.record_device_load(u64::from(width));
        }

        stats.record_instruction(InstructionCategory::Memory);
        Ok(())
    }

    fn execute_device_store(
        wave: &mut Wave,
        inst: &DecodedInstruction,
        device_memory: &mut DeviceMemory,
        stats: &mut ExecutionStats,
    ) -> Result<(), EmulatorError> {
        let width: u32 = match inst.modifier {
            m if m == MemWidth::U8 as u8 => 1,
            m if m == MemWidth::U16 as u8 => 2,
            m if m == MemWidth::U32 as u8 => 4,
            m if m == MemWidth::U64 as u8 => 8,
            m if m == MemWidth::U128 as u8 => 16,
            _ => 4,
        };

        for lane in 0..wave.wave_width {
            if !wave.is_thread_active(lane) {
                continue;
            }

            let thread = &wave.threads[lane as usize];
            let addr = u64::from(thread.read_register(inst.rs1));
            let value = thread.read_register(inst.rs2);

            match width {
                1 => device_memory.write_u8(addr, value.to_le_bytes()[0])?,
                2 => device_memory.write_u16(addr, Self::low_half(value))?,
                4 => device_memory.write_u32(addr, value)?,
                8 => {
                    let hi = thread.read_register(inst.rs2 + 1);
                    let val = (u64::from(hi) << 32) | u64::from(value);
                    device_memory.write_u64(addr, val)?;
                }
                16 => {
                    let w0 = value;
                    let w1 = thread.read_register(inst.rs2 + 1);
                    let w2 = thread.read_register(inst.rs2 + 2);
                    let w3 = thread.read_register(inst.rs2 + 3);
                    let val = u128::from(w0)
                        | (u128::from(w1) << 32)
                        | (u128::from(w2) << 64)
                        | (u128::from(w3) << 96);
                    device_memory.write_u128(addr, val)?;
                }
                _ => {}
            }

            stats.record_device_store(u64::from(width));
        }

        stats.record_instruction(InstructionCategory::Memory);
        Ok(())
    }

    fn execute_local_atomic(
        wave: &mut Wave,
        inst: &DecodedInstruction,
        local_memory: &mut LocalMemory,
        stats: &mut ExecutionStats,
    ) -> Result<(), EmulatorError> {
        let non_returning = inst.is_non_returning_atomic();

        for lane in 0..wave.wave_width {
            if !wave.is_thread_active(lane) {
                continue;
            }

            let thread = &mut wave.threads[lane as usize];
            let addr = thread.read_register(inst.rs1);
            let value = thread.read_register(inst.rs2);

            let old = match inst.modifier {
                m if m == AtomicOp::Add as u8 => local_memory.atomic_add(addr, value)?,
                m if m == AtomicOp::Sub as u8 => local_memory.atomic_sub(addr, value)?,
                m if m == AtomicOp::Min as u8 => local_memory.atomic_min(addr, value)?,
                m if m == AtomicOp::Max as u8 => local_memory.atomic_max(addr, value)?,
                m if m == AtomicOp::And as u8 => local_memory.atomic_and(addr, value)?,
                m if m == AtomicOp::Or as u8 => local_memory.atomic_or(addr, value)?,
                m if m == AtomicOp::Xor as u8 => local_memory.atomic_xor(addr, value)?,
                m if m == AtomicOp::Exchange as u8 => local_memory.atomic_exchange(addr, value)?,
                _ => {
                    let desired = thread.read_register(inst.rs3);
                    local_memory.atomic_cas(addr, value, desired)?
                }
            };

            if !non_returning {
                thread.write_register(inst.rd, old);
            }

            stats.atomic_ops += 1;
        }

        stats.record_instruction(InstructionCategory::Atomic);
        Ok(())
    }

    fn execute_device_atomic(
        wave: &mut Wave,
        inst: &DecodedInstruction,
        device_memory: &mut DeviceMemory,
        stats: &mut ExecutionStats,
    ) -> Result<(), EmulatorError> {
        let non_returning = inst.is_non_returning_atomic();

        for lane in 0..wave.wave_width {
            if !wave.is_thread_active(lane) {
                continue;
            }

            let thread = &mut wave.threads[lane as usize];
            let addr = u64::from(thread.read_register(inst.rs1));
            let value = thread.read_register(inst.rs2);

            let old = match inst.modifier {
                m if m == AtomicOp::Add as u8 => device_memory.atomic_add(addr, value)?,
                m if m == AtomicOp::Sub as u8 => device_memory.atomic_sub(addr, value)?,
                m if m == AtomicOp::Min as u8 => device_memory.atomic_min(addr, value)?,
                m if m == AtomicOp::Max as u8 => device_memory.atomic_max(addr, value)?,
                m if m == AtomicOp::And as u8 => device_memory.atomic_and(addr, value)?,
                m if m == AtomicOp::Or as u8 => device_memory.atomic_or(addr, value)?,
                m if m == AtomicOp::Xor as u8 => device_memory.atomic_xor(addr, value)?,
                m if m == AtomicOp::Exchange as u8 => device_memory.atomic_exchange(addr, value)?,
                _ => {
                    let desired = thread.read_register(inst.rs3);
                    device_memory.atomic_cas(addr, value, desired)?
                }
            };

            if !non_returning {
                thread.write_register(inst.rd, old);
            }

            stats.atomic_ops += 1;
        }

        stats.record_instruction(InstructionCategory::Atomic);
        Ok(())
    }

    fn execute_wave_op(wave: &mut Wave, inst: &DecodedInstruction) {
        if inst.is_wave_reduce() {
            let reduce_mod = inst.modifier - 8;
            match reduce_mod {
                m if m == WaveReduceType::PrefixSum as u8 => {
                    shuffle::wave_prefix_sum(wave, inst.rd, inst.rs1);
                }
                m if m == WaveReduceType::ReduceAdd as u8 => {
                    shuffle::wave_reduce_add(wave, inst.rd, inst.rs1);
                }
                m if m == WaveReduceType::ReduceMin as u8 => {
                    shuffle::wave_reduce_min(wave, inst.rd, inst.rs1);
                }
                m if m == WaveReduceType::ReduceMax as u8 => {
                    shuffle::wave_reduce_max(wave, inst.rd, inst.rs1);
                }
                _ => {}
            }
        } else {
            match inst.modifier {
                m if m == WaveOpType::Shuffle as u8 => {
                    shuffle::wave_shuffle(wave, inst.rd, inst.rs1, inst.rs2);
                }
                m if m == WaveOpType::ShuffleUp as u8 => {
                    shuffle::wave_shuffle_up(wave, inst.rd, inst.rs1, inst.rs2);
                }
                m if m == WaveOpType::ShuffleDown as u8 => {
                    shuffle::wave_shuffle_down(wave, inst.rd, inst.rs1, inst.rs2);
                }
                m if m == WaveOpType::ShuffleXor as u8 => {
                    shuffle::wave_shuffle_xor(wave, inst.rd, inst.rs1, inst.rs2);
                }
                m if m == WaveOpType::Broadcast as u8 => {
                    shuffle::wave_broadcast(wave, inst.rd, inst.rs1, inst.rs2);
                }
                m if m == WaveOpType::Ballot as u8 => {
                    shuffle::wave_ballot(wave, inst.rd, inst.rs1);
                }
                m if m == WaveOpType::Any as u8 => {
                    shuffle::wave_any(wave, inst.rd, inst.rs1);
                }
                m if m == WaveOpType::All as u8 => {
                    shuffle::wave_all(wave, inst.rd, inst.rs1);
                }
                _ => {}
            }
        }
    }

    fn execute_control(
        &mut self,
        wave: &mut Wave,
        inst: &DecodedInstruction,
        stats: &mut ExecutionStats,
    ) -> Result<ExecuteResult, EmulatorError> {
        stats.record_instruction(InstructionCategory::Control);

        if inst.is_sync_op() {
            return Ok(Self::execute_sync_op(wave, inst));
        }

        self.execute_control_flow(wave, inst, stats)
    }

    fn execute_sync_op(wave: &mut Wave, inst: &DecodedInstruction) -> ExecuteResult {
        let sync_mod = inst.modifier.saturating_sub(SYNC_MODIFIER_OFFSET);
        match sync_mod {
            m if m == SyncOp::Return as u8 => {
                if let Some(return_pc) = wave.pop_call() {
                    ExecuteResult::Jump(return_pc)
                } else {
                    ExecuteResult::Halt
                }
            }
            m if m == SyncOp::Halt as u8 => ExecuteResult::Halt,
            m if m == SyncOp::Barrier as u8 => ExecuteResult::Barrier,
            _ => ExecuteResult::Continue,
        }
    }

    fn execute_misc_op(wave: &mut Wave, inst: &DecodedInstruction) -> ExecuteResult {
        match inst.modifier {
            m if m == MiscOp::Mov as u8 => {
                for lane in 0..wave.wave_width {
                    if wave.is_thread_active(lane) {
                        let thread = &mut wave.threads[lane as usize];
                        let value = thread.read_register(inst.rs1);
                        thread.write_register(inst.rd, value);
                    }
                }
            }
            m if m == MiscOp::MovImm as u8 => {
                for lane in 0..wave.wave_width {
                    if wave.is_thread_active(lane) {
                        wave.threads[lane as usize].write_register(inst.rd, inst.immediate);
                    }
                }
            }
            m if m == MiscOp::MovSr as u8 => {
                for lane in 0..wave.wave_width {
                    if wave.is_thread_active(lane) {
                        let thread = &mut wave.threads[lane as usize];
                        let value = thread.read_special(inst.rs1);
                        thread.write_register(inst.rd, value);
                    }
                }
            }
            _ => {}
        }
        ExecuteResult::Continue
    }

    // The match keeps structured control flow transitions together.

    fn execute_control_flow(
        &mut self,
        wave: &mut Wave,
        inst: &DecodedInstruction,
        stats: &mut ExecutionStats,
    ) -> Result<ExecuteResult, EmulatorError> {
        match inst.modifier {
            m if m == ControlOp::If as u8 => {
                let pred_mask = Self::compute_predicate_mask(wave, inst.rs1, false);

                let then_mask = wave.active_mask & pred_mask;
                let else_mask = wave.active_mask & !pred_mask;

                if then_mask != wave.active_mask && else_mask != 0 {
                    stats.record_divergent_branch();
                }

                let (new_mask, _) = wave.control_flow.handle_if(wave.active_mask, pred_mask)?;
                wave.active_mask = new_mask;
                Ok(ExecuteResult::Continue)
            }
            m if m == ControlOp::Else as u8 => {
                let (new_mask, _) = wave.control_flow.handle_else(wave.active_mask)?;
                wave.active_mask = new_mask;
                Ok(ExecuteResult::Continue)
            }
            m if m == ControlOp::Endif as u8 => {
                let new_mask = wave.control_flow.handle_endif()?;
                wave.active_mask = new_mask;
                Ok(ExecuteResult::Continue)
            }
            m if m == ControlOp::Loop as u8 => {
                let body_start = wave.pc + inst.size;
                let new_mask = wave
                    .control_flow
                    .handle_loop(wave.active_mask, body_start)?;
                wave.active_mask = new_mask;
                Ok(ExecuteResult::Continue)
            }
            m if m == ControlOp::Break as u8 => {
                let pred_mask = Self::compute_predicate_mask(wave, inst.rs1, false);

                if self.trace.is_enabled() {
                    eprintln!(
                        "  BREAK: active_mask=0x{:x}, pred_mask=0x{:x}, pred_reg=p{}",
                        wave.active_mask, pred_mask, inst.rs1
                    );
                }

                let (new_mask, jump) = wave
                    .control_flow
                    .handle_break(wave.active_mask, pred_mask)?;
                wave.active_mask = new_mask;

                if self.trace.is_enabled() {
                    eprintln!("  BREAK: new_active_mask=0x{new_mask:x}, jump={jump:?}");
                }

                Ok(ExecuteResult::Continue)
            }
            m if m == ControlOp::Continue as u8 => {
                let pred_mask = Self::compute_predicate_mask(wave, inst.rs1, false);

                let (new_mask, jump) = wave
                    .control_flow
                    .handle_continue(wave.active_mask, pred_mask)?;
                wave.active_mask = new_mask;
                if let Some(target) = jump {
                    Ok(ExecuteResult::Jump(target))
                } else {
                    Ok(ExecuteResult::Continue)
                }
            }
            m if m == ControlOp::Endloop as u8 => {
                if self.trace.is_enabled() {
                    eprintln!("  ENDLOOP: active_mask=0x{:x}", wave.active_mask);
                }

                let (new_mask, jump) = wave.control_flow.handle_endloop(wave.active_mask)?;
                wave.active_mask = new_mask;

                if self.trace.is_enabled() {
                    eprintln!("  ENDLOOP: new_active_mask=0x{new_mask:x}, jump={jump:?}");
                }

                if let Some(target) = jump {
                    Ok(ExecuteResult::Jump(target))
                } else {
                    Ok(ExecuteResult::Continue)
                }
            }
            m if m == ControlOp::Call as u8 => {
                let return_pc = wave.pc + inst.size;
                wave.push_call(return_pc)
                    .map_err(|_| EmulatorError::StackOverflow {
                        kind: "call".into(),
                    })?;
                Ok(ExecuteResult::Jump(inst.immediate))
            }
            _ => Ok(ExecuteResult::Continue),
        }
    }

    fn execute_mma(
        wave: &mut Wave,
        inst: &DecodedInstruction,
        local_memory: &mut LocalMemory,
    ) -> Result<(), EmulatorError> {
        let mma_m = wave.threads[0].special_registers.mma_m;
        let mma_n = wave.threads[0].special_registers.mma_n;
        let mma_k = wave.threads[0].special_registers.mma_k;

        for lane in 0..wave.wave_width {
            if !wave.is_thread_active(lane) {
                continue;
            }
            let idx = lane as usize;

            match inst.modifier {
                0 => {
                    let frag_id = inst.rd;
                    let addr = wave.threads[idx].read_register(inst.rs1);
                    let stride = wave.threads[idx].read_register(inst.rs2);
                    let mut data = Vec::with_capacity((mma_m * mma_k) as usize);
                    for row in 0..mma_m {
                        for col in 0..mma_k {
                            let byte_addr = addr + row * stride + col * 4;
                            let bits = local_memory.read_u32(byte_addr)?;
                            data.push(f32::from_bits(bits));
                        }
                    }
                    wave.threads[idx]
                        .mma_fragments
                        .a_fragments
                        .insert(frag_id, data);
                }
                1 => {
                    let frag_id = inst.rd;
                    let addr = wave.threads[idx].read_register(inst.rs1);
                    let stride = wave.threads[idx].read_register(inst.rs2);
                    let mut data = Vec::with_capacity((mma_k * mma_n) as usize);
                    for row in 0..mma_k {
                        for col in 0..mma_n {
                            let byte_addr = addr + row * stride + col * 4;
                            let bits = local_memory.read_u32(byte_addr)?;
                            data.push(f32::from_bits(bits));
                        }
                    }
                    wave.threads[idx]
                        .mma_fragments
                        .b_fragments
                        .insert(frag_id, data);
                }
                2 => {
                    let addr = wave.threads[idx].read_register(inst.rd);
                    let stride = wave.threads[idx].read_register(inst.rs1);
                    let frag_id = inst.rs2;
                    let c_data = wave.threads[idx]
                        .mma_fragments
                        .c_fragments
                        .get(&frag_id)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; (mma_m * mma_n) as usize]);
                    for row in 0..mma_m {
                        for col in 0..mma_n {
                            let byte_addr = addr + row * stride + col * 4;
                            let val = c_data[(row * mma_n + col) as usize];
                            local_memory.write_u32(byte_addr, val.to_bits())?;
                        }
                    }
                }
                3 => {
                    let a_frag_id = inst.rd;
                    let b_frag_id = inst.rs1;
                    let c_frag_id = inst.rs2;
                    let a = wave.threads[idx]
                        .mma_fragments
                        .a_fragments
                        .get(&a_frag_id)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; (mma_m * mma_k) as usize]);
                    let b = wave.threads[idx]
                        .mma_fragments
                        .b_fragments
                        .get(&b_frag_id)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; (mma_k * mma_n) as usize]);
                    let c = wave.threads[idx]
                        .mma_fragments
                        .c_fragments
                        .entry(c_frag_id)
                        .or_insert_with(|| vec![0.0; (mma_m * mma_n) as usize]);
                    for i in 0..mma_m as usize {
                        for j in 0..mma_n as usize {
                            for k in 0..mma_k as usize {
                                c[i * mma_n as usize + j] +=
                                    a[i * mma_k as usize + k] * b[k * mma_n as usize + j];
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepResult {
    Continue,
    Halted,
    Barrier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecuteResult {
    Continue,
    Jump(u32),
    Halt,
    Barrier,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn encode_word0(opcode: u8, rd: u8, rs1: u8, modifier: u8, pred: u8) -> Vec<u8> {
        let word = ((u32::from(opcode) & 0xFF) << 24)
            | ((u32::from(rd) & 0xFF) << 16)
            | ((u32::from(rs1) & 0xFF) << 8)
            | ((u32::from(modifier) & 0x0F) << 4)
            | u32::from(pred & 0x07);
        word.to_le_bytes().to_vec()
    }

    fn encode_with_rs2(opcode: u8, rd: u8, rs1: u8, rs2: u8, modifier: u8, pred: u8) -> Vec<u8> {
        let mut code = encode_word0(opcode, rd, rs1, modifier, pred);
        let word1 = (u32::from(rs2) & 0xFF) << 24;
        code.extend_from_slice(&word1.to_le_bytes());
        code
    }

    fn encode_extended_imm(
        opcode: u8,
        rd: u8,
        rs1: u8,
        modifier: u8,
        pred: u8,
        imm: u32,
    ) -> Vec<u8> {
        let mut code = encode_word0(opcode, rd, rs1, modifier, pred);
        code.extend_from_slice(&imm.to_le_bytes());
        code
    }

    fn single_lane_wave() -> Wave {
        Wave::new(crate::wave::WaveConfig {
            wave_width: 1,
            register_count: 32,
            wave_id: 0,
            workgroup_id: [0; 3],
            workgroup_size: [1; 3],
            grid_size: [1; 3],
            base_thread_index: 0,
            total_threads_in_workgroup: 1,
            num_waves: 1,
        })
    }

    #[test]
    fn float_to_integer_saturates_and_truncates() {
        for (value, signed, unsigned) in [
            (f32::NAN, 0, 0),
            (f32::INFINITY, i32::MAX, u32::MAX),
            (f32::NEG_INFINITY, i32::MIN, 0),
            (-1.75, -1, 0),
            (1.75, 1, 1),
            (-0.0, 0, 0),
            (2_147_483_648.0, i32::MAX, 2_147_483_648),
            (4_294_967_296.0, i32::MAX, u32::MAX),
        ] {
            for (conversion, expected) in [
                (CvtType::I32F32, signed.cast_unsigned()),
                (CvtType::U32F32, unsigned),
            ] {
                let code = encode_word0(Opcode::Cvt as u8, 2, 1, conversion as u8, 0);
                let inst = Decoder::new(&code).decode_at(0).unwrap();
                let mut wave = single_lane_wave();
                wave.threads[0].write_register(1, value.to_bits());
                Executor::execute_convert(&mut wave, &inst);
                assert_eq!(wave.threads[0].read_register(2), expected);
            }
        }
    }

    #[test]
    fn float_comparison_preserves_nan_and_signed_zero() {
        for (a, b, equal, unequal) in [
            (0.0_f32, -0.0_f32, true, false),
            (f32::NAN, f32::NAN, false, true),
            (1.0, f32::from_bits(1.0_f32.to_bits() + 1), false, true),
            (f32::INFINITY, f32::INFINITY, true, false),
        ] {
            for (op, expected) in [(CmpOp::Eq, equal), (CmpOp::Ne, unequal)] {
                let code = encode_with_rs2(Opcode::Fcmp as u8, 1, 1, 2, op as u8, 0);
                let inst = Decoder::new(&code).decode_at(0).unwrap();
                let mut wave = single_lane_wave();
                wave.threads[0].write_register(1, a.to_bits());
                wave.threads[0].write_register(2, b.to_bits());
                Executor::execute_compare(&mut wave, &inst);
                assert_eq!(wave.threads[0].read_predicate(1), expected);
            }
        }
    }

    #[test]
    fn integer_to_float_rounds_to_nearest_even() {
        let code = encode_word0(Opcode::Cvt as u8, 2, 1, CvtType::F32U32 as u8, 0);
        let inst = Decoder::new(&code).decode_at(0).unwrap();
        let mut wave = single_lane_wave();
        for (value, expected) in [
            (16_777_217, 16_777_216.0_f32),
            (16_777_219, 16_777_220.0),
            (u32::MAX, 4_294_967_296.0),
        ] {
            wave.threads[0].write_register(1, value);
            Executor::execute_convert(&mut wave, &inst);
            assert_eq!(wave.threads[0].read_register(2), expected.to_bits());
        }
    }

    #[test]
    fn test_executor_iadd() {
        let code = encode_with_rs2(0x00, 3, 1, 2, 0, 0);
        let mut wave = Wave::new(crate::wave::WaveConfig {
            wave_width: 4,
            register_count: 32,
            wave_id: 0,
            workgroup_id: [0, 0, 0],
            workgroup_size: [4, 1, 1],
            grid_size: [1, 1, 1],
            base_thread_index: 0,
            total_threads_in_workgroup: 4,
            num_waves: 1,
        });

        for i in 0..4 {
            wave.threads[i].write_register(1, 10);
            wave.threads[i].write_register(2, 20);
        }

        let mut executor = Executor::new(&code, false, [0, 0, 0]);
        let mut local_memory = LocalMemory::new(1024);
        let mut device_memory = DeviceMemory::new(1024);
        let mut stats = ExecutionStats::new();

        executor
            .step(&mut wave, &mut local_memory, &mut device_memory, &mut stats)
            .unwrap();

        for i in 0..4 {
            assert_eq!(wave.threads[i].read_register(3), 30);
        }
    }

    #[test]
    fn test_executor_mov_imm() {
        let code = encode_extended_imm(0x41, 5, 0, MiscOp::MovImm as u8, 0, 0xDEAD_BEEF);
        let mut wave = Wave::new(crate::wave::WaveConfig {
            wave_width: 4,
            register_count: 32,
            wave_id: 0,
            workgroup_id: [0, 0, 0],
            workgroup_size: [4, 1, 1],
            grid_size: [1, 1, 1],
            base_thread_index: 0,
            total_threads_in_workgroup: 4,
            num_waves: 1,
        });

        let mut executor = Executor::new(&code, false, [0, 0, 0]);
        let mut local_memory = LocalMemory::new(1024);
        let mut device_memory = DeviceMemory::new(1024);
        let mut stats = ExecutionStats::new();

        executor
            .step(&mut wave, &mut local_memory, &mut device_memory, &mut stats)
            .unwrap();

        for i in 0..4 {
            assert_eq!(wave.threads[i].read_register(5), 0xDEAD_BEEF);
        }
    }

    #[test]
    fn test_executor_respects_active_mask() {
        let code = encode_with_rs2(0x00, 3, 1, 2, 0, 0);
        let mut wave = Wave::new(crate::wave::WaveConfig {
            wave_width: 4,
            register_count: 32,
            wave_id: 0,
            workgroup_id: [0, 0, 0],
            workgroup_size: [4, 1, 1],
            grid_size: [1, 1, 1],
            base_thread_index: 0,
            total_threads_in_workgroup: 4,
            num_waves: 1,
        });

        wave.active_mask = 0b0101;

        for i in 0..4 {
            wave.threads[i].write_register(1, 10);
            wave.threads[i].write_register(2, 20);
            wave.threads[i].write_register(3, 0);
        }

        let mut executor = Executor::new(&code, false, [0, 0, 0]);
        let mut local_memory = LocalMemory::new(1024);
        let mut device_memory = DeviceMemory::new(1024);
        let mut stats = ExecutionStats::new();

        executor
            .step(&mut wave, &mut local_memory, &mut device_memory, &mut stats)
            .unwrap();

        assert_eq!(wave.threads[0].read_register(3), 30);
        assert_eq!(wave.threads[1].read_register(3), 0);
        assert_eq!(wave.threads[2].read_register(3), 30);
        assert_eq!(wave.threads[3].read_register(3), 0);
    }
}
