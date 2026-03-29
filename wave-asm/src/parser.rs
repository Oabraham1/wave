// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Recursive descent parser. Converts token stream to AST (Program of Statements).
//!
//! Each line is one statement: directive, label, or instruction with optional
//! predicate and operands. Errors carry byte spans for diagnostic reporting.

use crate::ast::{
    Directive, Immediate, Instruction, Operand, Predicate, Program, Register, RegisterHalf,
    RegisterKind, Span, Spanned, Statement,
};
use crate::diagnostics::AssemblerError;
use crate::lexer::{SpannedToken, Token};
use crate::opcodes::{lookup_special_register, CacheHint, Scope, MAX_REGISTER_INDEX};

pub struct Parser {
    tokens: Vec<SpannedToken>,
    position: usize,
}

impl Parser {
    #[must_use]
    pub fn new(tokens: Vec<SpannedToken>) -> Self {
        Self {
            tokens,
            position: 0,
        }
    }

    pub fn parse(&mut self) -> Result<Program, AssemblerError> {
        let mut statements = Vec::new();

        while !self.is_at_end() {
            self.skip_newlines();
            if self.is_at_end() {
                break;
            }

            let stmt = self.parse_statement()?;
            statements.push(stmt);
        }

        Ok(Program { statements })
    }

    fn parse_statement(&mut self) -> Result<Spanned<Statement>, AssemblerError> {
        let start_span = self.current_span();

        match self.peek() {
            Some(Token::Directive(_)) => self.parse_directive(),
            Some(Token::Label(_)) => self.parse_label(),
            Some(Token::Predicate { .. } | Token::Identifier(_)) => self.parse_instruction(),
            Some(_) => Err(AssemblerError::UnexpectedToken {
                expected: "directive, label, or instruction".into(),
                span: start_span,
            }),
            None => Err(AssemblerError::UnexpectedToken {
                expected: "statement".into(),
                span: start_span,
            }),
        }
    }

    fn parse_directive(&mut self) -> Result<Spanned<Statement>, AssemblerError> {
        let start_span = self.current_span();
        let Token::Directive(name) = self.advance().token else {
            unreachable!()
        };

        let directive = match name.as_str() {
            "kernel" => {
                let kernel_name = self.expect_identifier()?;
                Directive::Kernel(kernel_name)
            }
            "end" => Directive::End,
            "registers" => {
                let count = self.expect_integer()? as u32;
                Directive::Registers(count)
            }
            "local_memory" => {
                let size = self.expect_integer()? as u32;
                Directive::LocalMemory(size)
            }
            "workgroup_size" => {
                let x = self.expect_integer()? as u32;
                self.expect_token(Token::Comma)?;
                let y = self.expect_integer()? as u32;
                self.expect_token(Token::Comma)?;
                let z = self.expect_integer()? as u32;
                Directive::WorkgroupSize(x, y, z)
            }
            _ => {
                return Err(AssemblerError::InvalidDirective { span: start_span });
            }
        };

        let end_span = self.previous_span();
        Ok(Spanned::new(
            Statement::Directive(directive),
            start_span.merge(end_span),
        ))
    }

    fn parse_label(&mut self) -> Result<Spanned<Statement>, AssemblerError> {
        let start_span = self.current_span();
        let Token::Label(name) = self.advance().token else {
            unreachable!()
        };

        self.expect_token(Token::Colon)?;

        let end_span = self.previous_span();
        Ok(Spanned::new(
            Statement::Label(name),
            start_span.merge(end_span),
        ))
    }

    fn parse_instruction(&mut self) -> Result<Spanned<Statement>, AssemblerError> {
        let start_span = self.current_span();

        let predicate = if matches!(self.peek(), Some(Token::Predicate { .. })) {
            Some(self.parse_predicate()?)
        } else {
            None
        };

        let mnemonic = self.expect_identifier()?;

        let mut cache_hint = None;
        if let Some(Token::Directive(hint_name)) = self.peek() {
            let hint_name = hint_name.clone();
            match hint_name.as_str() {
                "cached" => {
                    self.advance();
                    cache_hint = Some(CacheHint::Cached);
                }
                "uncached" => {
                    self.advance();
                    cache_hint = Some(CacheHint::Uncached);
                }
                "streaming" => {
                    self.advance();
                    cache_hint = Some(CacheHint::Streaming);
                }
                _ => {}
            }
        }

        let operands = self.parse_operands()?;

        let end_span = self.previous_span();
        Ok(Spanned::new(
            Statement::Instruction(Instruction {
                predicate,
                mnemonic,
                operands,
                cache_hint,
            }),
            start_span.merge(end_span),
        ))
    }

    fn parse_predicate(&mut self) -> Result<Predicate, AssemblerError> {
        let Token::Predicate { register, negated } = self.advance().token else {
            unreachable!()
        };

        let index = register
            .trim_start_matches('p')
            .parse::<u8>()
            .map_err(|_| AssemblerError::InvalidPredicate {
                span: self.previous_span(),
            })?;

        if index > 3 {
            return Err(AssemblerError::InvalidPredicateIndex {
                index,
                span: self.previous_span(),
            });
        }

        Ok(Predicate {
            register: index,
            negated,
        })
    }

    fn parse_operands(&mut self) -> Result<Vec<Spanned<Operand>>, AssemblerError> {
        let mut operands = Vec::new();

        if self.is_at_line_end() {
            return Ok(operands);
        }

        operands.push(self.parse_operand()?);

        while self.check(&Token::Comma) {
            self.advance();
            operands.push(self.parse_operand()?);
        }

        Ok(operands)
    }

    fn parse_operand(&mut self) -> Result<Spanned<Operand>, AssemblerError> {
        let span = self.current_span();

        match self.peek() {
            Some(Token::Register(reg)) => {
                let reg = reg.clone();
                self.advance();
                let register = self.parse_register_name(&reg, span)?;
                Ok(Spanned::new(Operand::Register(register), span))
            }
            Some(Token::PredicateReg(reg)) => {
                let reg = reg.clone();
                self.advance();
                let index = reg
                    .trim_start_matches('p')
                    .parse::<u8>()
                    .map_err(|_| AssemblerError::InvalidPredicate { span })?;
                if index > 3 {
                    return Err(AssemblerError::InvalidPredicateIndex { index, span });
                }
                Ok(Spanned::new(
                    Operand::Register(Register {
                        kind: RegisterKind::Predicate,
                        index,
                        half: None,
                    }),
                    span,
                ))
            }
            Some(Token::SpecialReg(reg)) => {
                let reg = reg.clone();
                self.advance();
                let index = lookup_special_register(&reg).ok_or_else(|| {
                    AssemblerError::InvalidSpecialRegister {
                        name: reg.clone(),
                        span,
                    }
                })?;
                Ok(Spanned::new(
                    Operand::Register(Register {
                        kind: RegisterKind::Special,
                        index,
                        half: None,
                    }),
                    span,
                ))
            }
            Some(Token::Integer(v)) => {
                let v = *v;
                self.advance();
                Ok(Spanned::new(
                    Operand::Immediate(Immediate::Integer(v)),
                    span,
                ))
            }
            Some(Token::Float(v)) => {
                let v = *v;
                self.advance();
                Ok(Spanned::new(Operand::Immediate(Immediate::Float(v)), span))
            }
            Some(Token::Identifier(name)) => {
                let name = name.clone();
                self.advance();
                let operand = self.parse_identifier_operand(&name, span)?;
                Ok(Spanned::new(operand, span))
            }
            Some(Token::Dot) => {
                self.advance();
                let scope_name = self.expect_identifier()?;
                let scope = match scope_name.as_str() {
                    "wave" => Scope::Wave,
                    "workgroup" => Scope::Workgroup,
                    "device" => Scope::Device,
                    "system" => Scope::System,
                    _ => {
                        return Err(AssemblerError::ExpectedScope { span });
                    }
                };
                Ok(Spanned::new(Operand::Scope(scope), span))
            }
            Some(Token::Directive(name)) => {
                let name = name.clone();
                self.advance();
                let scope = match name.as_str() {
                    "wave" => Scope::Wave,
                    "workgroup" => Scope::Workgroup,
                    "device" => Scope::Device,
                    "system" => Scope::System,
                    _ => {
                        return Err(AssemblerError::ExpectedScope { span });
                    }
                };
                Ok(Spanned::new(Operand::Scope(scope), span))
            }
            _ => Err(AssemblerError::UnexpectedToken {
                expected: "operand".into(),
                span,
            }),
        }
    }

    fn parse_register_name(&self, name: &str, span: Span) -> Result<Register, AssemblerError> {
        let (base, half) = if let Some(dot_pos) = name.find('.') {
            let (base, suffix) = name.split_at(dot_pos);
            let half = match &suffix[1..] {
                "lo" => Some(RegisterHalf::Lo),
                "hi" => Some(RegisterHalf::Hi),
                _ => {
                    return Err(AssemblerError::UnexpectedToken {
                        expected: "register half (.lo or .hi)".into(),
                        span,
                    });
                }
            };
            (base, half)
        } else {
            (name, None)
        };

        let index = base
            .trim_start_matches('r')
            .parse::<u8>()
            .map_err(|_| AssemblerError::ExpectedRegister { span })?;

        if index > MAX_REGISTER_INDEX {
            return Err(AssemblerError::InvalidRegisterIndex { index, span });
        }

        Ok(Register {
            kind: RegisterKind::General,
            index,
            half,
        })
    }

    fn parse_identifier_operand(&self, name: &str, _span: Span) -> Result<Operand, AssemblerError> {
        match name {
            "wave" => Ok(Operand::Scope(Scope::Wave)),
            "workgroup" => Ok(Operand::Scope(Scope::Workgroup)),
            "device" => Ok(Operand::Scope(Scope::Device)),
            "system" => Ok(Operand::Scope(Scope::System)),
            _ => Ok(Operand::Label(name.to_string())),
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.position).map(|t| &t.token)
    }

    fn advance(&mut self) -> SpannedToken {
        let token = self.tokens[self.position].clone();
        self.position += 1;
        token
    }

    fn check(&self, expected: &Token) -> bool {
        self.peek()
            .is_some_and(|t| std::mem::discriminant(t) == std::mem::discriminant(expected))
    }

    fn is_at_end(&self) -> bool {
        matches!(self.peek(), Some(Token::Eof) | None)
    }

    fn is_at_line_end(&self) -> bool {
        matches!(self.peek(), Some(Token::Newline | Token::Eof) | None)
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), Some(Token::Newline)) {
            self.advance();
        }
    }

    fn current_span(&self) -> Span {
        self.tokens
            .get(self.position)
            .map_or(Span::new(0, 0), |t| t.span)
    }

    fn previous_span(&self) -> Span {
        self.tokens
            .get(self.position.saturating_sub(1))
            .map_or(Span::new(0, 0), |t| t.span)
    }

    fn expect_token(&mut self, expected: Token) -> Result<SpannedToken, AssemblerError> {
        if self.check(&expected) {
            Ok(self.advance())
        } else {
            Err(AssemblerError::UnexpectedToken {
                expected: format!("{expected:?}"),
                span: self.current_span(),
            })
        }
    }

    fn expect_identifier(&mut self) -> Result<String, AssemblerError> {
        match self.peek() {
            Some(Token::Identifier(name)) => {
                let name = name.clone();
                self.advance();
                Ok(name)
            }
            _ => Err(AssemblerError::UnexpectedToken {
                expected: "identifier".into(),
                span: self.current_span(),
            }),
        }
    }

    fn expect_integer(&mut self) -> Result<i64, AssemblerError> {
        match self.peek() {
            Some(Token::Integer(v)) => {
                let v = *v;
                self.advance();
                Ok(v)
            }
            _ => Err(AssemblerError::UnexpectedToken {
                expected: "integer".into(),
                span: self.current_span(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    fn parse(source: &str) -> Result<Program, AssemblerError> {
        let tokens = Lexer::new(source).tokenize()?;
        Parser::new(tokens).parse()
    }

    #[test]
    fn test_parser_simple_instruction() {
        let program = parse("iadd r0, r1, r2").unwrap();
        assert_eq!(program.statements.len(), 1);

        let Statement::Instruction(inst) = &program.statements[0].node else {
            panic!("expected instruction");
        };
        assert_eq!(inst.mnemonic, "iadd");
        assert_eq!(inst.operands.len(), 3);
    }

    #[test]
    fn test_parser_predicated_instruction() {
        let program = parse("@p0 iadd r0, r1, r2").unwrap();
        let Statement::Instruction(inst) = &program.statements[0].node else {
            panic!("expected instruction");
        };

        assert!(inst.predicate.is_some());
        let pred = inst.predicate.as_ref().unwrap();
        assert_eq!(pred.register, 0);
        assert!(!pred.negated);
    }

    #[test]
    fn test_parser_negated_predicate() {
        let program = parse("@!p1 fadd r0, r1, r2").unwrap();
        let Statement::Instruction(inst) = &program.statements[0].node else {
            panic!("expected instruction");
        };

        let pred = inst.predicate.as_ref().unwrap();
        assert_eq!(pred.register, 1);
        assert!(pred.negated);
    }

    #[test]
    fn test_parser_kernel_directive() {
        let program = parse(".kernel my_kernel").unwrap();
        let Statement::Directive(Directive::Kernel(name)) = &program.statements[0].node else {
            panic!("expected kernel directive");
        };
        assert_eq!(name, "my_kernel");
    }

    #[test]
    fn test_parser_registers_directive() {
        let program = parse(".registers 64").unwrap();
        let Statement::Directive(Directive::Registers(count)) = &program.statements[0].node else {
            panic!("expected registers directive");
        };
        assert_eq!(*count, 64);
    }

    #[test]
    fn test_parser_label() {
        let program = parse("loop_start:").unwrap();
        let Statement::Label(name) = &program.statements[0].node else {
            panic!("expected label");
        };
        assert_eq!(name, "loop_start");
    }

    #[test]
    fn test_parser_immediate_operand() {
        let program = parse("mov_imm r0, 0xFF").unwrap();
        let Statement::Instruction(inst) = &program.statements[0].node else {
            panic!("expected instruction");
        };

        let Operand::Immediate(Immediate::Integer(v)) = &inst.operands[1].node else {
            panic!("expected immediate");
        };
        assert_eq!(*v, 255);
    }

    #[test]
    fn test_parser_special_register() {
        let program = parse("mov r0, sr_thread_id_x").unwrap();
        let Statement::Instruction(inst) = &program.statements[0].node else {
            panic!("expected instruction");
        };

        let Operand::Register(reg) = &inst.operands[1].node else {
            panic!("expected register");
        };
        assert_eq!(reg.kind, RegisterKind::Special);
        assert_eq!(reg.index, 0);
    }

    #[test]
    fn test_parser_register_half() {
        let program = parse("hadd r0.lo, r1.hi, r2.lo").unwrap();
        let Statement::Instruction(inst) = &program.statements[0].node else {
            panic!("expected instruction");
        };

        let Operand::Register(r0) = &inst.operands[0].node else {
            panic!("expected register");
        };
        assert_eq!(r0.half, Some(RegisterHalf::Lo));

        let Operand::Register(r1) = &inst.operands[1].node else {
            panic!("expected register");
        };
        assert_eq!(r1.half, Some(RegisterHalf::Hi));
    }

    #[test]
    fn test_parser_scope_operand() {
        let program = parse("fence_acquire .workgroup").unwrap();
        let Statement::Instruction(inst) = &program.statements[0].node else {
            panic!("expected instruction");
        };

        let Operand::Scope(scope) = &inst.operands[0].node else {
            panic!("expected scope");
        };
        assert_eq!(*scope, Scope::Workgroup);
    }

    #[test]
    fn test_parser_label_operand() {
        let program = parse("call my_function").unwrap();
        let Statement::Instruction(inst) = &program.statements[0].node else {
            panic!("expected instruction");
        };

        let Operand::Label(label) = &inst.operands[0].node else {
            panic!("expected label");
        };
        assert_eq!(label, "my_function");
    }

    #[test]
    fn test_parser_multiple_statements() {
        let program = parse(
            ".kernel test
iadd r0, r1, r2
isub r3, r4, r5
.end",
        )
        .unwrap();

        assert_eq!(program.statements.len(), 4);
        assert!(matches!(
            &program.statements[0].node,
            Statement::Directive(Directive::Kernel(_))
        ));
        assert!(matches!(
            &program.statements[1].node,
            Statement::Instruction(_)
        ));
        assert!(matches!(
            &program.statements[2].node,
            Statement::Instruction(_)
        ));
        assert!(matches!(
            &program.statements[3].node,
            Statement::Directive(Directive::End)
        ));
    }

    #[test]
    fn test_parser_cache_hint() {
        let program = parse("device_load_u32.uncached r0, r1").unwrap();
        let Statement::Instruction(inst) = &program.statements[0].node else {
            panic!("expected instruction");
        };

        assert_eq!(inst.cache_hint, Some(CacheHint::Uncached));
    }

    #[test]
    fn test_parser_select_with_predicate_operand() {
        let program = parse("select r0, p1, r2, r3").unwrap();
        let Statement::Instruction(inst) = &program.statements[0].node else {
            panic!("expected instruction");
        };

        let Operand::Register(pred_reg) = &inst.operands[1].node else {
            panic!("expected predicate register");
        };
        assert_eq!(pred_reg.kind, RegisterKind::Predicate);
        assert_eq!(pred_reg.index, 1);
    }

    #[test]
    fn test_parser_workgroup_size_directive() {
        let program = parse(".workgroup_size 256, 1, 1").unwrap();
        let Statement::Directive(Directive::WorkgroupSize(x, y, z)) = &program.statements[0].node
        else {
            panic!("expected workgroup_size directive");
        };
        assert_eq!(*x, 256);
        assert_eq!(*y, 1);
        assert_eq!(*z, 1);
    }

    #[test]
    fn test_parser_invalid_predicate_index() {
        let result = parse("@p5 iadd r0, r1, r2");
        assert!(result.is_err());
    }

    #[test]
    fn test_parser_invalid_register_index() {
        let result = parse("iadd r0, r1, r99");
        assert!(result.is_err());
    }
}
