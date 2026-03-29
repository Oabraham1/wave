// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Compiler error types covering all stages of compilation.
//!
//! Errors carry enough context for rich error reporting including
//! source locations, expected vs found types, and suggestions.

use thiserror::Error;

/// Source location in the input file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceLoc {
    /// Line number (1-based).
    pub line: u32,
    /// Column number (1-based).
    pub col: u32,
}

/// Compiler error covering all compilation stages.
#[derive(Debug, Error)]
pub enum CompileError {
    /// Type mismatch in expression.
    #[error("type mismatch: expected {expected}, found {found}")]
    TypeMismatch {
        /// Expected type.
        expected: String,
        /// Found type.
        found: String,
    },

    /// Reference to undefined variable.
    #[error("undefined variable: {name}")]
    UndefinedVariable {
        /// Variable name.
        name: String,
    },

    /// Parse error in frontend.
    #[error("parse error: {message}")]
    ParseError {
        /// Error message.
        message: String,
    },

    /// Unsupported language construct.
    #[error("unsupported: {message}")]
    Unsupported {
        /// Description of unsupported construct.
        message: String,
    },

    /// Internal compiler error (bug).
    #[error("internal error: {message}")]
    InternalError {
        /// Error message.
        message: String,
    },

    /// I/O error.
    #[error("I/O error: {message}")]
    IoError {
        /// Error message.
        message: String,
    },

    /// Code generation error.
    #[error("codegen error: {message}")]
    CodegenError {
        /// Error message.
        message: String,
    },

    /// Register allocation failure.
    #[error("register allocation failed: {message}")]
    RegAllocError {
        /// Error message.
        message: String,
    },
}

impl From<std::io::Error> for CompileError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError {
            message: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CompileError::TypeMismatch {
            expected: "i32".into(),
            found: "f32".into(),
        };
        assert_eq!(
            err.to_string(),
            "type mismatch: expected i32, found f32"
        );
    }

    #[test]
    fn test_undefined_variable_error() {
        let err = CompileError::UndefinedVariable {
            name: "foo".into(),
        };
        assert_eq!(err.to_string(), "undefined variable: foo");
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let compile_err: CompileError = io_err.into();
        assert!(compile_err.to_string().contains("file not found"));
    }
}
