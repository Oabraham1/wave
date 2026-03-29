// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Compiler diagnostics: error types, source mapping, and error reporting.
//!
//! Provides structured error types and formatting utilities for
//! presenting compiler errors to users with source context.

pub mod error;
pub mod report;
pub mod source_map;

pub use error::CompileError;
pub use report::format_error;
pub use source_map::SourceMap;
