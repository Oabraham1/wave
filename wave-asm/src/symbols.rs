// Copyright (c) 2026 Ojima Abraham. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

// Two-pass symbol table. Pass 1 collects label definitions and their byte
// offsets. Pass 2 patches forward references in call instructions. Duplicate
// labels and unresolved references are reported as errors.

use crate::ast::{Program, Span, Statement};
use crate::diagnostics::AssemblerError;
use crate::opcodes::lookup_mnemonic;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Symbol {
    pub offset: u32,
    pub span: Span,
}

pub struct SymbolTable {
    symbols: HashMap<String, Symbol>,
}

impl SymbolTable {
    #[must_use]
    pub fn new() -> Self {
        Self {
            symbols: HashMap::new(),
        }
    }

    pub fn define(&mut self, name: String, offset: u32, span: Span) -> Result<(), AssemblerError> {
        if self.symbols.contains_key(&name) {
            return Err(AssemblerError::DuplicateLabel {
                label: name,
                span,
            });
        }
        self.symbols.insert(name, Symbol { offset, span });
        Ok(())
    }

    #[must_use]
    pub fn resolve(&self, name: &str) -> Option<u32> {
        self.symbols.get(name).map(|s| s.offset)
    }

    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.symbols.contains_key(name)
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

pub fn build_symbol_table(program: &Program) -> Result<SymbolTable, AssemblerError> {
    let mut symbols = SymbolTable::new();
    let mut current_offset: u32 = 0;

    for stmt in &program.statements {
        match &stmt.node {
            Statement::Label(name) => {
                symbols.define(name.clone(), current_offset, stmt.span)?;
            }
            Statement::Instruction(inst) => {
                if let Some(sig) = lookup_mnemonic(&inst.mnemonic) {
                    current_offset += if sig.extended { 8 } else { 4 };
                } else {
                    return Err(AssemblerError::UnknownInstruction {
                        mnemonic: inst.mnemonic.clone(),
                        span: stmt.span,
                    });
                }
            }
            Statement::Directive(_) => {}
        }
    }

    Ok(symbols)
}

pub fn validate_label_references(
    program: &Program,
    symbols: &SymbolTable,
) -> Result<(), AssemblerError> {
    for stmt in &program.statements {
        if let Statement::Instruction(inst) = &stmt.node {
            for operand in &inst.operands {
                if let crate::ast::Operand::Label(label) = &operand.node {
                    if !symbols.contains(label) {
                        return Err(AssemblerError::UndefinedLabel {
                            label: label.clone(),
                            span: operand.span,
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn parse(source: &str) -> Program {
        let tokens = Lexer::new(source).tokenize().unwrap();
        Parser::new(tokens).parse().unwrap()
    }

    #[test]
    fn test_symbols_single_label() {
        let program = parse(
            "start:
iadd r0, r1, r2",
        );
        let symbols = build_symbol_table(&program).unwrap();

        assert_eq!(symbols.resolve("start"), Some(0));
    }

    #[test]
    fn test_symbols_multiple_labels() {
        let program = parse(
            "start:
iadd r0, r1, r2
middle:
isub r3, r4, r5
end:",
        );
        let symbols = build_symbol_table(&program).unwrap();

        assert_eq!(symbols.resolve("start"), Some(0));
        assert_eq!(symbols.resolve("middle"), Some(4));
        assert_eq!(symbols.resolve("end"), Some(8));
    }

    #[test]
    fn test_symbols_extended_instruction() {
        let program = parse(
            "start:
mov_imm r0, 0x12345678
next:",
        );
        let symbols = build_symbol_table(&program).unwrap();

        assert_eq!(symbols.resolve("start"), Some(0));
        assert_eq!(symbols.resolve("next"), Some(8));
    }

    #[test]
    fn test_symbols_duplicate_label() {
        let program = parse(
            "start:
iadd r0, r1, r2
start:",
        );
        let result = build_symbol_table(&program);

        assert!(matches!(result, Err(AssemblerError::DuplicateLabel { .. })));
    }

    #[test]
    fn test_symbols_forward_reference_valid() {
        let program = parse(
            "call target
nop
target:
return",
        );
        let symbols = build_symbol_table(&program).unwrap();
        let result = validate_label_references(&program, &symbols);

        assert!(result.is_ok());
    }

    #[test]
    fn test_symbols_undefined_label() {
        let program = parse("call undefined_label");
        let symbols = build_symbol_table(&program).unwrap();
        let result = validate_label_references(&program, &symbols);

        assert!(matches!(
            result,
            Err(AssemblerError::UndefinedLabel { .. })
        ));
    }
}
