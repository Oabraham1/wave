// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! WAVE compiler: compiles high-level GPU kernel code to WAVE ISA binaries.
//!
//! Supports Python, Rust, C++, and TypeScript frontends. Includes a full
//! optimization pipeline with SSA-based analysis and graph coloring register
//! allocation.

pub mod analysis;
pub mod diagnostics;
pub mod driver;
pub mod emit;
pub mod frontend;
pub mod hir;
pub mod lir;
pub mod lowering;
pub mod mir;
pub mod optimize;
pub mod regalloc;

pub use diagnostics::CompileError;
pub use driver::{compile_kernel, compile_source, CompilerConfig, Language, OptLevel};
