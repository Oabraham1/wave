// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Compiler driver: configuration, session management, and pipeline orchestration.
//!
//! The driver module is the top-level entry point that coordinates all
//! compilation stages from source code to WAVE binary.

pub mod config;
pub mod pipeline;
pub mod session;

pub use config::{CompilerConfig, Language, OptLevel};
pub use pipeline::{compile_kernel, compile_source};
pub use session::Session;
