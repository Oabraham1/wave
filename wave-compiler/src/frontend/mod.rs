// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Language frontends for parsing source code into HIR.
//!
//! Each frontend parses a specific language (Python, Rust, C++, TypeScript)
//! and produces a language-independent HIR kernel definition.

pub mod cpp;
pub mod python;
pub mod rust;
pub mod typescript;

use crate::diagnostics::CompileError;
use crate::driver::config::Language;
use crate::hir::kernel::Kernel;

/// Parse source code in the given language and produce an HIR kernel.
///
/// # Errors
///
/// Returns `CompileError` if parsing fails.
pub fn parse(source: &str, language: Language) -> Result<Kernel, CompileError> {
    match language {
        Language::Python => python::parse_python(source),
        Language::Rust => rust::parse_rust(source),
        Language::Cpp => cpp::parse_cpp(source),
        Language::TypeScript => typescript::parse_typescript(source),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_frontend() {
        let source = r#"
@kernel
def test(n: u32):
    x = 42
"#;
        let kernel = parse(source, Language::Python).unwrap();
        assert_eq!(kernel.name, "test");
    }

    #[test]
    fn test_rust_frontend() {
        let source = r#"
#[kernel]
fn test(n: u32) {
    let x = 42;
}
"#;
        let kernel = parse(source, Language::Rust).unwrap();
        assert_eq!(kernel.name, "test");
    }
}
