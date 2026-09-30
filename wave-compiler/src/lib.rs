// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Compile Python, Rust, C++, and TypeScript kernels to WAVE binaries.

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
