// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Runtime error types for the WAVE runtime.
//!
//! Unifies errors from all stages of the pipeline: compilation, backend
//! translation, device detection, memory management, and kernel launch.

use thiserror::Error;

/// Errors that can occur during WAVE runtime operations.
#[derive(Debug, Error)]
pub enum RuntimeError {
    /// Kernel source compilation failed.
    #[error("compilation error: {0}")]
    Compile(String),

    /// Backend translation failed.
    #[error("backend error: {0}")]
    Backend(String),

    /// GPU device detection failed.
    #[error("device error: {0}")]
    Device(String),

    /// Memory operation failed.
    #[error("memory error: {0}")]
    Memory(String),

    /// Kernel launch failed.
    #[error("launch error: {0}")]
    Launch(String),

    /// Emulator execution failed.
    #[error("emulator error: {0}")]
    Emulator(String),

    /// I/O error during subprocess or file operations.
    #[error("I/O error: {0}")]
    Io(String),

    /// Invalid argument provided.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

impl From<wave_compiler::CompileError> for RuntimeError {
    fn from(e: wave_compiler::CompileError) -> Self {
        Self::Compile(e.to_string())
    }
}

impl From<wave_emu::EmulatorError> for RuntimeError {
    fn from(e: wave_emu::EmulatorError) -> Self {
        Self::Emulator(e.to_string())
    }
}

impl From<std::io::Error> for RuntimeError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}
