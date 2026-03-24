// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Abstract syntax tree types. The parser produces a Program containing
//!
//! Statement nodes (labels, instructions, directives). Each node carries a
//! Span for error reporting. Operands distinguish registers, immediates, labels.

use crate::opcodes::{CacheHint, Scope};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    #[must_use]
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    #[must_use]
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }
}

#[derive(Debug, Clone)]
pub struct Program {
    pub statements: Vec<Spanned<Statement>>,
}

#[derive(Debug, Clone)]
pub enum Statement {
    Label(String),
    Instruction(Instruction),
    Directive(Directive),
}

#[derive(Debug, Clone)]
pub enum Directive {
    Kernel(String),
    End,
    Registers(u32),
    LocalMemory(u32),
    WorkgroupSize(u32, u32, u32),
}

#[derive(Debug, Clone)]
pub struct Instruction {
    pub predicate: Option<Predicate>,
    pub mnemonic: String,
    pub operands: Vec<Spanned<Operand>>,
    pub cache_hint: Option<CacheHint>,
}

#[derive(Debug, Clone)]
pub struct Predicate {
    pub register: u8,
    pub negated: bool,
}

#[derive(Debug, Clone)]
pub enum Operand {
    Register(Register),
    Immediate(Immediate),
    Label(String),
    Scope(Scope),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterKind {
    General,
    Special,
    Predicate,
}

#[derive(Debug, Clone)]
pub struct Register {
    pub kind: RegisterKind,
    pub index: u8,
    pub half: Option<RegisterHalf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterHalf {
    Lo,
    Hi,
}

#[derive(Debug, Clone, Copy)]
pub enum Immediate {
    Integer(i64),
    Float(f64),
}

impl Immediate {
    #[must_use]
    #[allow(clippy::checked_conversions)]
    pub fn as_u32(self) -> Option<u32> {
        match self {
            Self::Integer(v) => u32::try_from(v).ok().or_else(|| {
                i32::try_from(v).ok().map(|i| i as u32)
            }),
            Self::Float(f) => Some(f.to_bits() as u32),
        }
    }

    #[must_use]
    pub fn as_i32(self) -> Option<i32> {
        match self {
            Self::Integer(v) => i32::try_from(v).ok(),
            Self::Float(_) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KernelMetadata {
    pub name: String,
    pub register_count: u32,
    pub local_memory_size: u32,
    pub workgroup_size: [u32; 3],
    pub code_offset: u32,
    pub code_size: u32,
}

impl Default for KernelMetadata {
    fn default() -> Self {
        Self {
            name: String::new(),
            register_count: 32,
            local_memory_size: 0,
            workgroup_size: [0, 0, 0], // 0 = use CLI value
            code_offset: 0,
            code_size: 0,
        }
    }
}
