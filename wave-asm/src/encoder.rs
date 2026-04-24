// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Instruction encoder. Converts AST instructions to 32-bit or 64-bit machine words.
//!
//! Handles operand encoding, predication bits, and resolves label references via
//! the symbol table. Returns `EncodedInstruction` (single or extended format).

use crate::ast::{
    Immediate, Instruction, Operand, Predicate, Register, RegisterKind, Span, Spanned,
};
use crate::diagnostics::AssemblerError;
use crate::opcodes::{
    lookup_mnemonic, ControlOp, InstructionSignature, MiscOp, Opcode, OperandKind, Scope, SyncOp,
    WaveOpType, EXTENDED_RS2_SHIFT, EXTENDED_RS3_SHIFT, EXTENDED_RS4_SHIFT, EXTENDED_SCOPE_SHIFT,
    MODIFIER_MASK, MODIFIER_SHIFT, OPCODE_SHIFT, PRED_NEG_SHIFT, PRED_REG_MASK, PRED_REG_SHIFT,
    RD_SHIFT, RS1_SHIFT,
};
use crate::symbols::SymbolTable;

#[derive(Debug, Clone)]
pub struct EncodedInstruction {
    pub word0: u32,
    pub word1: Option<u32>,
}

impl EncodedInstruction {
    #[must_use]
    pub fn single(word: u32) -> Self {
        Self {
            word0: word,
            word1: None,
        }
    }

    #[must_use]
    pub fn extended(word0: u32, word1: u32) -> Self {
        Self {
            word0,
            word1: Some(word1),
        }
    }

    #[must_use]
    pub fn size_bytes(&self) -> usize {
        if self.word1.is_some() {
            8
        } else {
            4
        }
    }
}

pub struct Encoder<'a> {
    symbols: &'a SymbolTable,
    current_offset: u32,
}

impl<'a> Encoder<'a> {
    #[must_use]
    pub fn new(symbols: &'a SymbolTable) -> Self {
        Self {
            symbols,
            current_offset: 0,
        }
    }

    pub fn encode_instruction(
        &mut self,
        inst: &Instruction,
        span: Span,
    ) -> Result<EncodedInstruction, AssemblerError> {
        let sig =
            lookup_mnemonic(&inst.mnemonic).ok_or_else(|| AssemblerError::UnknownInstruction {
                mnemonic: inst.mnemonic.clone(),
                span,
            })?;

        self.validate_operand_count(inst, sig, span)?;

        let encoded = self.encode_with_signature(inst, sig, span)?;
        self.current_offset += encoded.size_bytes() as u32;
        Ok(encoded)
    }

    fn validate_operand_count(
        &self,
        inst: &Instruction,
        sig: &InstructionSignature,
        span: Span,
    ) -> Result<(), AssemblerError> {
        let expected = sig.operands.len();
        let got = inst.operands.len();

        let has_optional = sig
            .operands
            .iter()
            .any(|o| matches!(o, OperandKind::OptionalRd));

        if has_optional {
            if got < expected - 1 || got > expected {
                return Err(AssemblerError::InvalidOperandCount {
                    expected,
                    got,
                    span,
                });
            }
        } else if got != expected {
            return Err(AssemblerError::InvalidOperandCount {
                expected,
                got,
                span,
            });
        }

        Ok(())
    }

    fn encode_with_signature(
        &self,
        inst: &Instruction,
        sig: &InstructionSignature,
        span: Span,
    ) -> Result<EncodedInstruction, AssemblerError> {
        let opcode = sig.opcode.as_u8();
        let mut word0 = u32::from(opcode) << OPCODE_SHIFT;

        word0 |= self.encode_predicate(&inst.predicate);

        if let Some(modifier) = sig.modifier {
            word0 |= (u32::from(modifier) & MODIFIER_MASK) << MODIFIER_SHIFT;
        }

        match sig.opcode {
            Opcode::LocalAtomic | Opcode::DeviceAtomic => {
                self.encode_atomic(inst, sig, word0, span)
            }
            Opcode::Misc => self.encode_misc(inst, sig, word0, span),
            Opcode::Control => self.encode_control(inst, sig, word0, span),
            Opcode::WaveOp => self.encode_wave_op(inst, sig, word0, span),
            Opcode::Select => self.encode_select(inst, word0, span),
            _ if sig.extended => self.encode_extended(inst, sig, word0, span),
            _ => self.encode_base(inst, sig, word0, span),
        }
    }

    fn encode_predicate(&self, pred: &Option<Predicate>) -> u32 {
        match pred {
            Some(p) => {
                let reg = u32::from(p.register) & PRED_REG_MASK;
                let neg = u32::from(p.negated);
                (reg << PRED_REG_SHIFT) | (neg << PRED_NEG_SHIFT)
            }
            None => 0,
        }
    }

    fn encode_base(
        &self,
        inst: &Instruction,
        sig: &InstructionSignature,
        mut word0: u32,
        span: Span,
    ) -> Result<EncodedInstruction, AssemblerError> {
        let mut word1: u32 = 0;
        let mut needs_extended = false;

        for (i, (kind, operand)) in sig.operands.iter().zip(inst.operands.iter()).enumerate() {
            match kind {
                OperandKind::Rd => {
                    let reg = self.expect_register(operand)?;
                    word0 |= u32::from(reg) << RD_SHIFT;
                }
                OperandKind::Rs1 => {
                    let reg = self.expect_register(operand)?;
                    word0 |= u32::from(reg) << RS1_SHIFT;
                }
                OperandKind::Rs2 => {
                    let reg = self.expect_register(operand)?;
                    word1 |= u32::from(reg) << EXTENDED_RS2_SHIFT;
                    needs_extended = true;
                }
                OperandKind::Pd => {
                    let pred = self.expect_predicate(operand)?;
                    word0 |= u32::from(pred) << RD_SHIFT;
                }
                OperandKind::PdSrc => {
                    let pred = self.expect_predicate(operand)?;
                    if i == 0 {
                        word0 |= u32::from(pred) << RS1_SHIFT;
                    } else {
                        word1 |= u32::from(pred) << EXTENDED_RS2_SHIFT;
                        needs_extended = true;
                    }
                }
                OperandKind::Scope => {
                    let scope = self.expect_scope(operand)?;
                    word1 |= (scope as u32) << EXTENDED_SCOPE_SHIFT;
                    needs_extended = true;
                }
                _ => {
                    return Err(AssemblerError::UnexpectedToken {
                        expected: "valid operand".into(),
                        span,
                    });
                }
            }
        }

        if needs_extended {
            Ok(EncodedInstruction::extended(word0, word1))
        } else {
            Ok(EncodedInstruction::single(word0))
        }
    }

    fn encode_extended(
        &self,
        inst: &Instruction,
        sig: &InstructionSignature,
        mut word0: u32,
        _span: Span,
    ) -> Result<EncodedInstruction, AssemblerError> {
        let mut word1: u32 = 0;

        for (i, (kind, operand)) in sig.operands.iter().zip(inst.operands.iter()).enumerate() {
            match kind {
                OperandKind::Rd => {
                    let reg = self.expect_register(operand)?;
                    word0 |= u32::from(reg) << RD_SHIFT;
                }
                OperandKind::Rs1 => {
                    let reg = self.expect_register(operand)?;
                    word0 |= u32::from(reg) << RS1_SHIFT;
                }
                OperandKind::Rs2 => {
                    let reg = self.expect_register(operand)?;
                    word1 |= u32::from(reg) << EXTENDED_RS2_SHIFT;
                }
                OperandKind::Rs3 => {
                    let reg = self.expect_register(operand)?;
                    word1 |= u32::from(reg) << EXTENDED_RS3_SHIFT;
                }
                OperandKind::Rs4 => {
                    let reg = self.expect_register(operand)?;
                    word1 |= u32::from(reg) << EXTENDED_RS4_SHIFT;
                }
                OperandKind::Imm32 => {
                    let imm = self.expect_immediate(operand)?;
                    word1 = imm;
                }
                OperandKind::Pd => {
                    let pred = self.expect_predicate(operand)?;
                    word0 |= u32::from(pred) << RD_SHIFT;
                }
                OperandKind::PdSrc => {
                    let pred = self.expect_predicate(operand)?;
                    if i == 0 {
                        word0 |= u32::from(pred) << RS1_SHIFT;
                    } else {
                        word1 |= u32::from(pred) << EXTENDED_RS2_SHIFT;
                    }
                }
                OperandKind::Scope => {
                    let scope = self.expect_scope(operand)?;
                    word1 |= (scope as u32) << EXTENDED_SCOPE_SHIFT;
                }
                _ => {}
            }
        }

        Ok(EncodedInstruction::extended(word0, word1))
    }

    fn encode_atomic(
        &self,
        inst: &Instruction,
        sig: &InstructionSignature,
        mut word0: u32,
        span: Span,
    ) -> Result<EncodedInstruction, AssemblerError> {
        let has_optional = sig
            .operands
            .iter()
            .any(|o| matches!(o, OperandKind::OptionalRd));
        let expected_with_rd = sig.operands.len();
        let is_non_returning = has_optional && inst.operands.len() < expected_with_rd;

        let mut operand_idx = 0;
        let is_cas = sig.extended && sig.operands.iter().any(|o| matches!(o, OperandKind::Rs3));
        let mut word1: u32 = 0;

        for kind in sig.operands {
            match kind {
                OperandKind::OptionalRd => {
                    if !is_non_returning {
                        let reg = self.expect_register(&inst.operands[operand_idx])?;
                        word0 |= u32::from(reg) << RD_SHIFT;
                        operand_idx += 1;
                    }
                }
                OperandKind::Rs1 => {
                    let reg = self.expect_register(&inst.operands[operand_idx])?;
                    word0 |= u32::from(reg) << RS1_SHIFT;
                    operand_idx += 1;
                }
                OperandKind::Rs2 => {
                    let reg = self.expect_register(&inst.operands[operand_idx])?;
                    word1 |= u32::from(reg) << EXTENDED_RS2_SHIFT;
                    operand_idx += 1;
                }
                OperandKind::Rs3 if is_cas => {
                    let reg = self.expect_register(&inst.operands[operand_idx])?;
                    word1 |= u32::from(reg) << EXTENDED_RS3_SHIFT;
                    operand_idx += 1;
                }
                OperandKind::Scope => {
                    let scope = self.expect_scope(&inst.operands[operand_idx])?;
                    word1 |= (scope as u32) << EXTENDED_SCOPE_SHIFT;
                    operand_idx += 1;
                }
                _ => {
                    return Err(AssemblerError::UnexpectedToken {
                        expected: "atomic operand".into(),
                        span,
                    });
                }
            }
        }

        // All atomics now need extended word for rs2
        Ok(EncodedInstruction::extended(word0, word1))
    }

    fn encode_control(
        &self,
        inst: &Instruction,
        sig: &InstructionSignature,
        mut word0: u32,
        span: Span,
    ) -> Result<EncodedInstruction, AssemblerError> {
        let modifier = sig.modifier.unwrap_or(0);

        if modifier >= 8 {
            let sync_mod = modifier - 8;
            match sync_mod {
                op if op == SyncOp::FenceAcquire as u8
                    || op == SyncOp::FenceRelease as u8
                    || op == SyncOp::FenceAcqRel as u8 =>
                {
                    let mut word1: u32 = 0;
                    if !inst.operands.is_empty() {
                        let scope = self.expect_scope(&inst.operands[0])?;
                        word1 |= (scope as u32) << EXTENDED_SCOPE_SHIFT;
                    }
                    Ok(EncodedInstruction::extended(word0, word1))
                }
                _ => Ok(EncodedInstruction::single(word0)),
            }
        } else {
            match modifier {
                op if op == ControlOp::If as u8
                    || op == ControlOp::Break as u8
                    || op == ControlOp::Continue as u8 =>
                {
                    if !inst.operands.is_empty() {
                        let pred = self.expect_predicate(&inst.operands[0])?;
                        word0 |= u32::from(pred) << RS1_SHIFT;
                    }
                    Ok(EncodedInstruction::single(word0))
                }
                op if op == ControlOp::Call as u8 => {
                    let label = self.expect_label(&inst.operands[0])?;
                    let target = self.symbols.resolve(&label).ok_or_else(|| {
                        AssemblerError::UndefinedLabel {
                            label: label.clone(),
                            span,
                        }
                    })?;
                    Ok(EncodedInstruction::extended(word0, target))
                }
                _ => Ok(EncodedInstruction::single(word0)),
            }
        }
    }

    fn encode_wave_op(
        &self,
        inst: &Instruction,
        sig: &InstructionSignature,
        mut word0: u32,
        _span: Span,
    ) -> Result<EncodedInstruction, AssemblerError> {
        let modifier = sig.modifier.unwrap_or(0);

        if modifier == WaveOpType::Ballot as u8 {
            let rd = self.expect_register(&inst.operands[0])?;
            let pred = self.expect_predicate(&inst.operands[1])?;
            word0 |= u32::from(rd) << RD_SHIFT;
            word0 |= u32::from(pred) << RS1_SHIFT;
            Ok(EncodedInstruction::single(word0))
        } else if modifier == WaveOpType::Any as u8 || modifier == WaveOpType::All as u8 {
            let pd = self.expect_predicate(&inst.operands[0])?;
            let ps = self.expect_predicate(&inst.operands[1])?;
            word0 |= u32::from(pd) << RD_SHIFT;
            word0 |= u32::from(ps) << RS1_SHIFT;
            Ok(EncodedInstruction::single(word0))
        } else if modifier >= 8 {
            // Reduce ops: rd, rs1 only - no rs2
            let rd = self.expect_register(&inst.operands[0])?;
            let rs1 = self.expect_register(&inst.operands[1])?;
            word0 |= u32::from(rd) << RD_SHIFT;
            word0 |= u32::from(rs1) << RS1_SHIFT;
            Ok(EncodedInstruction::single(word0))
        } else {
            // Shuffle/broadcast variants: rd, rs1 in base, rs2 in extended
            let rd = self.expect_register(&inst.operands[0])?;
            let rs1 = self.expect_register(&inst.operands[1])?;
            word0 |= u32::from(rd) << RD_SHIFT;
            word0 |= u32::from(rs1) << RS1_SHIFT;
            let mut word1: u32 = 0;
            if inst.operands.len() > 2 {
                let rs2 = self.expect_register(&inst.operands[2])?;
                word1 |= u32::from(rs2) << EXTENDED_RS2_SHIFT;
            }
            Ok(EncodedInstruction::extended(word0, word1))
        }
    }

    fn encode_select(
        &self,
        inst: &Instruction,
        mut word0: u32,
        _span: Span,
    ) -> Result<EncodedInstruction, AssemblerError> {
        let rd = self.expect_register(&inst.operands[0])?;
        let pd = self.expect_predicate(&inst.operands[1])?;
        let rs1 = self.expect_register(&inst.operands[2])?;
        let rs2 = self.expect_register(&inst.operands[3])?;

        word0 |= u32::from(rd) << RD_SHIFT;
        word0 |= u32::from(pd) << RS1_SHIFT;

        let word1 = (u32::from(rs1) << EXTENDED_RS2_SHIFT) | (u32::from(rs2) << EXTENDED_RS3_SHIFT);

        Ok(EncodedInstruction::extended(word0, word1))
    }

    fn encode_misc(
        &self,
        inst: &Instruction,
        sig: &InstructionSignature,
        mut word0: u32,
        _span: Span,
    ) -> Result<EncodedInstruction, AssemblerError> {
        let modifier = sig.modifier.unwrap_or(0);

        let effective_modifier = if modifier == MiscOp::Mov as u8 && inst.operands.len() > 1 {
            if let Operand::Register(Register {
                kind: RegisterKind::Special,
                ..
            }) = &inst.operands[1].node
            {
                MiscOp::MovSr as u8
            } else {
                modifier
            }
        } else {
            modifier
        };

        word0 &= !((MODIFIER_MASK) << MODIFIER_SHIFT);
        word0 |= (u32::from(effective_modifier) & MODIFIER_MASK) << MODIFIER_SHIFT;

        match effective_modifier {
            m if m == MiscOp::Mov as u8 => {
                let rd = self.expect_register(&inst.operands[0])?;
                let rs1 = self.expect_register(&inst.operands[1])?;
                word0 |= u32::from(rd) << RD_SHIFT;
                word0 |= u32::from(rs1) << RS1_SHIFT;
                Ok(EncodedInstruction::single(word0))
            }
            m if m == MiscOp::MovImm as u8 => {
                let rd = self.expect_register(&inst.operands[0])?;
                let imm = self.expect_immediate(&inst.operands[1])?;
                word0 |= u32::from(rd) << RD_SHIFT;
                Ok(EncodedInstruction::extended(word0, imm))
            }
            m if m == MiscOp::MovSr as u8 => {
                let rd = self.expect_register(&inst.operands[0])?;
                let sr = self.expect_special_register(&inst.operands[1])?;
                word0 |= u32::from(rd) << RD_SHIFT;
                word0 |= u32::from(sr) << RS1_SHIFT;
                Ok(EncodedInstruction::single(word0))
            }
            _ => Ok(EncodedInstruction::single(word0)),
        }
    }

    fn expect_special_register(&self, operand: &Spanned<Operand>) -> Result<u8, AssemblerError> {
        match &operand.node {
            Operand::Register(Register {
                kind: RegisterKind::Special,
                index,
                ..
            }) => Ok(*index),
            _ => Err(AssemblerError::ExpectedRegister { span: operand.span }),
        }
    }

    fn expect_register(&self, operand: &Spanned<Operand>) -> Result<u8, AssemblerError> {
        match &operand.node {
            Operand::Register(reg) => {
                if reg.kind == RegisterKind::Predicate {
                    return Err(AssemblerError::ExpectedRegister { span: operand.span });
                }
                Ok(reg.index)
            }
            _ => Err(AssemblerError::ExpectedRegister { span: operand.span }),
        }
    }

    fn expect_predicate(&self, operand: &Spanned<Operand>) -> Result<u8, AssemblerError> {
        match &operand.node {
            Operand::Register(Register {
                kind: RegisterKind::Predicate,
                index,
                ..
            }) => Ok(*index),
            _ => Err(AssemblerError::ExpectedRegister { span: operand.span }),
        }
    }

    fn expect_immediate(&self, operand: &Spanned<Operand>) -> Result<u32, AssemblerError> {
        match &operand.node {
            Operand::Immediate(imm) => imm.as_u32().ok_or(AssemblerError::ImmediateOutOfRange {
                value: match imm {
                    Immediate::Integer(v) => *v,
                    Immediate::Float(_) => 0,
                },
                span: operand.span,
            }),
            _ => Err(AssemblerError::ExpectedImmediate { span: operand.span }),
        }
    }

    fn expect_scope(&self, operand: &Spanned<Operand>) -> Result<Scope, AssemblerError> {
        match &operand.node {
            Operand::Scope(s) => Ok(*s),
            _ => Err(AssemblerError::ExpectedScope { span: operand.span }),
        }
    }

    fn expect_label(&self, operand: &Spanned<Operand>) -> Result<String, AssemblerError> {
        match &operand.node {
            Operand::Label(l) => Ok(l.clone()),
            _ => Err(AssemblerError::ExpectedLabel { span: operand.span }),
        }
    }
}

pub fn encode_instruction_standalone(
    inst: &Instruction,
    span: Span,
) -> Result<EncodedInstruction, AssemblerError> {
    let symbols = SymbolTable::new();
    let mut encoder = Encoder::new(&symbols);
    encoder.encode_instruction(inst, span)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::opcodes::{EXTENDED_SCOPE_MASK, PRED_NEG_MASK};
    use crate::parser::Parser;

    fn encode(source: &str) -> Result<EncodedInstruction, AssemblerError> {
        let tokens = Lexer::new(source).tokenize()?;
        let program = Parser::new(tokens).parse()?;

        let crate::ast::Statement::Instruction(inst) = &program.statements[0].node else {
            panic!("expected instruction");
        };

        encode_instruction_standalone(inst, program.statements[0].span)
    }

    #[test]
    fn test_encoder_iadd() {
        let encoded = encode("iadd r1, r2, r3").unwrap();
        assert!(encoded.word1.is_some());

        let opcode = (encoded.word0 >> OPCODE_SHIFT) & 0xFF;
        assert_eq!(opcode, Opcode::Iadd as u32);

        let rd = (encoded.word0 >> RD_SHIFT) & 0xFF;
        assert_eq!(rd, 1);

        let rs1 = (encoded.word0 >> RS1_SHIFT) & 0xFF;
        assert_eq!(rs1, 2);

        let rs2 = (encoded.word1.unwrap() >> EXTENDED_RS2_SHIFT) & 0xFF;
        assert_eq!(rs2, 3);
    }

    #[test]
    fn test_encoder_mov_imm() {
        let encoded = encode("mov_imm r5, 0x12345678").unwrap();
        assert!(encoded.word1.is_some());

        let opcode = (encoded.word0 >> OPCODE_SHIFT) & 0xFF;
        assert_eq!(opcode, Opcode::Misc as u32);

        let rd = (encoded.word0 >> RD_SHIFT) & 0xFF;
        assert_eq!(rd, 5);

        assert_eq!(encoded.word1.unwrap(), 0x1234_5678);
    }

    #[test]
    fn test_encoder_predicated() {
        let encoded = encode("@p1 iadd r0, r1, r2").unwrap();
        let opcode = (encoded.word0 >> OPCODE_SHIFT) & 0xFF;
        assert_eq!(opcode, Opcode::Iadd as u32);
        let pred_reg = (encoded.word0 >> PRED_REG_SHIFT) & PRED_REG_MASK;
        assert_eq!(pred_reg, 1);
        let pred_neg = (encoded.word0 >> PRED_NEG_SHIFT) & PRED_NEG_MASK;
        assert_eq!(pred_neg, 0);
    }

    #[test]
    fn test_encoder_negated_predicate() {
        let encoded = encode("@!p2 fadd r0, r1, r2").unwrap();
        let opcode = (encoded.word0 >> OPCODE_SHIFT) & 0xFF;
        assert_eq!(opcode, Opcode::Fadd as u32);
        let pred_reg = (encoded.word0 >> PRED_REG_SHIFT) & PRED_REG_MASK;
        assert_eq!(pred_reg, 2);
        let pred_neg = (encoded.word0 >> PRED_NEG_SHIFT) & PRED_NEG_MASK;
        assert_eq!(pred_neg, 1);
    }

    #[test]
    fn test_encoder_fma_extended() {
        let encoded = encode("fma r0, r1, r2, r3").unwrap();
        assert!(encoded.word1.is_some());

        let opcode = (encoded.word0 >> OPCODE_SHIFT) & 0xFF;
        assert_eq!(opcode, Opcode::Fma as u32);

        let rs2 = (encoded.word1.unwrap() >> EXTENDED_RS2_SHIFT) & 0xFF;
        assert_eq!(rs2, 2);

        let rs3 = (encoded.word1.unwrap() >> EXTENDED_RS3_SHIFT) & 0xFF;
        assert_eq!(rs3, 3);
    }

    #[test]
    fn test_encoder_local_load() {
        let encoded = encode("local_load_u32 r0, r1").unwrap();

        let opcode = (encoded.word0 >> OPCODE_SHIFT) & 0xFF;
        assert_eq!(opcode, Opcode::LocalLoad as u32);

        let modifier = (encoded.word0 >> MODIFIER_SHIFT) & 0x0F;
        assert_eq!(modifier, 2);
    }

    #[test]
    fn test_encoder_fence() {
        let encoded = encode("fence_acquire .workgroup").unwrap();
        let word1 = encoded.word1.unwrap();
        let scope = (word1 >> EXTENDED_SCOPE_SHIFT) & EXTENDED_SCOPE_MASK;
        assert_eq!(scope, Scope::Workgroup as u32);
    }

    #[test]
    fn test_encoder_barrier() {
        let encoded = encode("barrier").unwrap();

        let opcode = (encoded.word0 >> OPCODE_SHIFT) & 0xFF;
        assert_eq!(opcode, Opcode::Control as u32);

        let modifier = (encoded.word0 >> MODIFIER_SHIFT) & 0x0F;
        assert_eq!(modifier, SyncOp::Barrier as u32 + 8);
    }

    #[test]
    fn test_encoder_nop() {
        let encoded = encode("nop").unwrap();

        let opcode = (encoded.word0 >> OPCODE_SHIFT) & 0xFF;
        assert_eq!(opcode, Opcode::Control as u32);
    }

    #[test]
    fn test_encoder_icmp() {
        let encoded = encode("icmp_lt p0, r1, r2").unwrap();

        let opcode = (encoded.word0 >> OPCODE_SHIFT) & 0xFF;
        assert_eq!(opcode, Opcode::Icmp as u32);

        let pd = (encoded.word0 >> RD_SHIFT) & 0xFF;
        assert_eq!(pd, 0);
    }

    #[test]
    fn test_encoder_wave_shuffle() {
        let encoded = encode("wave_shuffle r0, r1, r2").unwrap();

        let opcode = (encoded.word0 >> OPCODE_SHIFT) & 0xFF;
        assert_eq!(opcode, Opcode::WaveOp as u32);

        let modifier = (encoded.word0 >> MODIFIER_SHIFT) & 0x0F;
        assert_eq!(modifier, WaveOpType::Shuffle as u32);
    }

    #[test]
    fn test_encoder_unknown_instruction() {
        let result = encode("unknown_op r0, r1, r2");
        assert!(matches!(
            result,
            Err(AssemblerError::UnknownInstruction { .. })
        ));
    }

    #[test]
    fn test_encoder_wrong_operand_count() {
        let result = encode("iadd r0, r1");
        assert!(matches!(
            result,
            Err(AssemblerError::InvalidOperandCount { .. })
        ));
    }
}
