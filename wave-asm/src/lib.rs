// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Library root. Public API exposes `assemble()` and `assemble_with_options()` for
//!
//! converting .wave source text to .wbin binary. Coordinates lexer, parser,
//! symbol resolution, instruction encoding, and output generation.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::unused_self,
    clippy::needless_pass_by_value,
    clippy::unnecessary_wraps,
    clippy::ref_option,
    clippy::if_not_else,
    clippy::struct_excessive_bools,
    clippy::too_many_lines,
    clippy::must_use_candidate,
    clippy::too_many_arguments,
    clippy::redundant_closure_for_method_calls
)]

pub mod ast;
pub mod diagnostics;
pub mod encoder;
pub mod lexer;
pub mod opcodes;
pub mod output;
pub mod parser;
pub mod symbols;

#[cfg(test)]
mod tests;

use ast::{Directive, Program, Span, Statement};
use diagnostics::{AssemblerError, AssemblerWarning, DiagnosticEmitter};
use encoder::Encoder;
use lexer::Lexer;
use output::WbinWriter;
use parser::Parser;
use symbols::{build_symbol_table, validate_label_references};

pub struct AssemblerResult {
    pub binary: Vec<u8>,
    pub warnings: Vec<AssemblerWarning>,
}

#[derive(Default)]
pub struct AssemblerOptions {
    pub strip_symbols: bool,
}

pub fn assemble(source: &str, source_name: &str) -> Result<AssemblerResult, AssemblerError> {
    assemble_with_options(source, source_name, &AssemblerOptions::default())
}

pub fn assemble_with_options(
    source: &str,
    _source_name: &str,
    options: &AssemblerOptions,
) -> Result<AssemblerResult, AssemblerError> {
    let tokens = Lexer::new(source).tokenize()?;
    let program = Parser::new(tokens).parse()?;

    let mut warnings = Vec::new();
    validate_program(&program, &mut warnings)?;

    let symbols = build_symbol_table(&program)?;
    validate_label_references(&program, &symbols)?;

    let mut writer = WbinWriter::new();
    writer.set_strip_symbols(options.strip_symbols);
    let mut encoder = Encoder::new(&symbols);
    let mut in_kernel = false;

    for stmt in &program.statements {
        match &stmt.node {
            Statement::Directive(directive) => match directive {
                Directive::Kernel(name) => {
                    if in_kernel {
                        return Err(AssemblerError::UnclosedKernel {
                            name: name.clone(),
                            span: stmt.span,
                        });
                    }
                    writer.begin_kernel(name.clone());
                    in_kernel = true;
                }
                Directive::End => {
                    writer.end_kernel();
                    in_kernel = false;
                }
                Directive::Registers(count) => {
                    writer.set_register_count(*count);
                }
                Directive::LocalMemory(size) => {
                    writer.set_local_memory_size(*size);
                }
                Directive::WorkgroupSize(x, y, z) => {
                    writer.set_workgroup_size(*x, *y, *z);
                }
            },
            Statement::Label(_) => {}
            Statement::Instruction(inst) => {
                if !in_kernel {
                    return Err(AssemblerError::InstructionOutsideKernel { span: stmt.span });
                }
                let instruction = encoder.encode_instruction(inst, stmt.span)?;
                writer.write_instruction(&instruction);
            }
        }
    }

    let mut binary = Vec::new();
    writer.finish(&mut binary)?;

    Ok(AssemblerResult { binary, warnings })
}

fn validate_program(
    program: &Program,
    _warnings: &mut Vec<AssemblerWarning>,
) -> Result<(), AssemblerError> {
    let mut kernel_stack: Vec<(String, Span)> = Vec::new();

    for stmt in &program.statements {
        match &stmt.node {
            Statement::Directive(Directive::Kernel(name)) => {
                if let Some((prev_name, prev_span)) = kernel_stack.last() {
                    return Err(AssemblerError::UnclosedKernel {
                        name: prev_name.clone(),
                        span: *prev_span,
                    });
                }
                kernel_stack.push((name.clone(), stmt.span));
            }
            Statement::Directive(Directive::End) => {
                if kernel_stack.is_empty() {
                    return Err(AssemblerError::UnexpectedToken {
                        expected: ".kernel before .end".into(),
                        span: stmt.span,
                    });
                }
                kernel_stack.pop();
            }
            Statement::Directive(Directive::Registers(_count)) => {}
            _ => {}
        }
    }

    if let Some((name, span)) = kernel_stack.pop() {
        return Err(AssemblerError::UnclosedKernel { name, span });
    }

    Ok(())
}

pub fn print_diagnostics(
    errors: &[AssemblerError],
    warnings: &[AssemblerWarning],
    source_name: &str,
    source: &str,
) {
    let emitter = DiagnosticEmitter::new(source_name, source);
    let mut stderr = std::io::stderr();

    for error in errors {
        emitter.emit_error(error, &mut stderr);
    }

    for warning in warnings {
        emitter.emit_warning(warning, &mut stderr);
    }
}
