// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Error and warning types with source spans. Uses ariadne for pretty-printed
//!
//! diagnostics with color coding and source context. All assembler errors carry
//! byte offsets to enable precise error location reporting.

use crate::ast::Span;
use ariadne::{Color, Label, Report, ReportKind, Source};
use std::io::Write;
use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum AssemblerError {
    #[error("unexpected character '{char}'")]
    UnexpectedCharacter { char: char, span: Span },

    #[error("invalid directive")]
    InvalidDirective { span: Span },

    #[error("invalid predicate syntax")]
    InvalidPredicate { span: Span },

    #[error("invalid special register '{name}'")]
    InvalidSpecialRegister { name: String, span: Span },

    #[error("invalid number literal")]
    InvalidNumber { span: Span },

    #[error("unexpected token")]
    UnexpectedToken { expected: String, span: Span },

    #[error("unknown instruction '{mnemonic}'")]
    UnknownInstruction { mnemonic: String, span: Span },

    #[error("invalid operand count: expected {expected}, got {got}")]
    InvalidOperandCount {
        expected: usize,
        got: usize,
        span: Span,
    },

    #[error("invalid register index {index}: must be 0-31")]
    InvalidRegisterIndex { index: u8, span: Span },

    #[error("invalid predicate register index {index}: must be 0-3")]
    InvalidPredicateIndex { index: u8, span: Span },

    #[error("immediate value {value} out of range")]
    ImmediateOutOfRange { value: i64, span: Span },

    #[error("undefined label '{label}'")]
    UndefinedLabel { label: String, span: Span },

    #[error("duplicate label '{label}'")]
    DuplicateLabel { label: String, span: Span },

    #[error("expected register operand")]
    ExpectedRegister { span: Span },

    #[error("expected immediate operand")]
    ExpectedImmediate { span: Span },

    #[error("expected label operand")]
    ExpectedLabel { span: Span },

    #[error("expected scope specifier (.wave, .workgroup, .device, .system)")]
    ExpectedScope { span: Span },

    #[error("kernel '{name}' not closed with .end")]
    UnclosedKernel { name: String, span: Span },

    #[error("instruction outside kernel")]
    InstructionOutsideKernel { span: Span },

    #[error("I/O error: {message}")]
    IoError { message: String },
}

impl AssemblerError {
    #[must_use]
    pub fn span(&self) -> Option<Span> {
        match self {
            Self::UnexpectedCharacter { span, .. }
            | Self::InvalidDirective { span }
            | Self::InvalidPredicate { span }
            | Self::InvalidSpecialRegister { span, .. }
            | Self::InvalidNumber { span }
            | Self::UnexpectedToken { span, .. }
            | Self::UnknownInstruction { span, .. }
            | Self::InvalidOperandCount { span, .. }
            | Self::InvalidRegisterIndex { span, .. }
            | Self::InvalidPredicateIndex { span, .. }
            | Self::ImmediateOutOfRange { span, .. }
            | Self::UndefinedLabel { span, .. }
            | Self::DuplicateLabel { span, .. }
            | Self::ExpectedRegister { span }
            | Self::ExpectedImmediate { span }
            | Self::ExpectedLabel { span }
            | Self::ExpectedScope { span }
            | Self::UnclosedKernel { span, .. }
            | Self::InstructionOutsideKernel { span } => Some(*span),
            Self::IoError { .. } => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum AssemblerWarning {
    RegisterCountExceeds32 { count: u32, span: Span },
}

pub struct DiagnosticEmitter<'a> {
    source_name: &'a str,
    source: &'a str,
}

impl<'a> DiagnosticEmitter<'a> {
    #[must_use]
    pub fn new(source_name: &'a str, source: &'a str) -> Self {
        Self {
            source_name,
            source,
        }
    }

    pub fn emit_error<W: Write>(&self, error: &AssemblerError, writer: &mut W) {
        let Some(span) = error.span() else {
            let _ = writeln!(writer, "error: {error}");
            return;
        };

        let report = Report::build(ReportKind::Error, self.source_name, span.start)
            .with_message(error.to_string())
            .with_label(
                Label::new((self.source_name, span.start..span.end))
                    .with_color(Color::Red)
                    .with_message(self.error_hint(error)),
            )
            .finish();

        let _ = report.write((self.source_name, Source::from(self.source)), writer);
    }

    pub fn emit_warning<W: Write>(&self, warning: &AssemblerWarning, writer: &mut W) {
        match warning {
            AssemblerWarning::RegisterCountExceeds32 { count, span } => {
                let report = Report::build(ReportKind::Warning, self.source_name, span.start)
                    .with_message(format!(
                        "register count {count} exceeds 5-bit encoding limit (32)"
                    ))
                    .with_label(
                        Label::new((self.source_name, span.start..span.end))
                            .with_color(Color::Yellow)
                            .with_message("capped to 32 for v0.1"),
                    )
                    .finish();

                let _ = report.write((self.source_name, Source::from(self.source)), writer);
            }
        }
    }

    fn error_hint(&self, error: &AssemblerError) -> String {
        match error {
            AssemblerError::UnexpectedCharacter { char, .. } => {
                format!("unexpected '{char}'")
            }
            AssemblerError::InvalidDirective { .. } => "invalid directive".into(),
            AssemblerError::InvalidPredicate { .. } => "expected @pN or @!pN".into(),
            AssemblerError::InvalidSpecialRegister { name, .. } => {
                format!("unknown special register '{name}'")
            }
            AssemblerError::InvalidNumber { .. } => "invalid number".into(),
            AssemblerError::UnexpectedToken { expected, .. } => {
                format!("expected {expected}")
            }
            AssemblerError::UnknownInstruction { mnemonic, .. } => {
                format!("'{mnemonic}' is not a valid instruction")
            }
            AssemblerError::InvalidOperandCount { expected, got, .. } => {
                format!("expected {expected} operands, found {got}")
            }
            AssemblerError::InvalidRegisterIndex { index, .. } => {
                format!("register r{index} out of range")
            }
            AssemblerError::InvalidPredicateIndex { index, .. } => {
                format!("predicate p{index} out of range")
            }
            AssemblerError::ImmediateOutOfRange { value, .. } => {
                format!("{value} doesn't fit in field")
            }
            AssemblerError::UndefinedLabel { label, .. } => {
                format!("'{label}' not defined")
            }
            AssemblerError::DuplicateLabel { label, .. } => {
                format!("'{label}' already defined")
            }
            AssemblerError::ExpectedRegister { .. } => "expected register".into(),
            AssemblerError::ExpectedImmediate { .. } => "expected immediate".into(),
            AssemblerError::ExpectedLabel { .. } => "expected label".into(),
            AssemblerError::ExpectedScope { .. } => "expected scope".into(),
            AssemblerError::UnclosedKernel { name, .. } => {
                format!("kernel '{name}' needs .end")
            }
            AssemblerError::InstructionOutsideKernel { .. } => {
                "must be inside .kernel block".into()
            }
            AssemblerError::IoError { message } => message.clone(),
        }
    }
}
