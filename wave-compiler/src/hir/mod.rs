// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! High-Level Intermediate Representation (HIR) for WAVE GPU kernels.
//!
//! HIR is the first language-independent representation produced by all
//! frontends. It preserves structured control flow and high-level types.

pub mod expr;
pub mod kernel;
pub mod stmt;
pub mod types;
pub mod validate;

pub use expr::{BinOp, BuiltinFunc, Dimension, Expr, Literal, MemoryScope, ShuffleMode, UnaryOp};
pub use kernel::{Kernel, KernelAttributes, KernelParam};
pub use stmt::Stmt;
pub use types::{AddressSpace, Type};
pub use validate::validate_kernel;
