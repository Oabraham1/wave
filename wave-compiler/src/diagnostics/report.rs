// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Error reporting with source context.
//!
//! Formats compiler errors with the relevant source line and
//! a caret pointing to the error location.

use std::fmt::Write;

use super::error::{CompileError, SourceLoc};
use super::source_map::SourceMap;

/// Format a compile error with source context for display to the user.
#[must_use]
pub fn format_error(
    error: &CompileError,
    source_map: Option<&SourceMap>,
    loc: Option<SourceLoc>,
) -> String {
    let mut out = String::new();

    if let Some(loc) = loc {
        let _ = write!(out, "error at {}:{}: ", loc.line, loc.col);
    } else {
        out.push_str("error: ");
    }

    out.push_str(&error.to_string());
    out.push('\n');

    if let (Some(map), Some(loc)) = (source_map, loc) {
        if let Some(line) = map.get_line(loc.line) {
            let _ = writeln!(out, " {} | {}", loc.line, line);
            let padding = format!("{}", loc.line).len() + 3 + (loc.col as usize - 1);
            let _ = writeln!(out, "{}^", " ".repeat(padding));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_error_with_location() {
        let err = CompileError::UndefinedVariable { name: "x".into() };
        let src = "let y = x + 1".to_string();
        let map = SourceMap::new(src);
        let loc = SourceLoc { line: 1, col: 9 };
        let output = format_error(&err, Some(&map), Some(loc));
        assert!(output.contains("error at 1:9"));
        assert!(output.contains("undefined variable: x"));
        assert!(output.contains("let y = x + 1"));
    }

    #[test]
    fn test_format_error_without_location() {
        let err = CompileError::InternalError {
            message: "something went wrong".into(),
        };
        let output = format_error(&err, None, None);
        assert!(output.contains("error: internal error"));
        assert!(output.contains("something went wrong"));
    }
}
