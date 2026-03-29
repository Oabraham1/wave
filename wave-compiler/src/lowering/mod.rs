// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! IR lowering passes: HIR→MIR and MIR→LIR.
//!
//! These passes transform between the compiler's intermediate representations,
//! flattening structured control flow and performing instruction selection.

pub mod hir_to_mir;
pub mod mir_to_lir;

pub use hir_to_mir::lower_kernel;
pub use mir_to_lir::lower_function;
