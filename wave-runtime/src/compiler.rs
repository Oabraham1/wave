// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Kernel compilation wrapper for the WAVE runtime.
//!
//! Wraps `wave_compiler::compile_source` to compile high-level kernel source
//! code into WAVE binary (.wbin) format. Supports Python, Rust, C++, and
//! TypeScript source languages.

use crate::error::RuntimeError;
use wave_compiler::{CompilerConfig, Language, OptLevel};

/// Compile kernel source code to a WAVE binary.
///
/// # Errors
///
/// Returns `RuntimeError::Compile` if the source cannot be parsed or compiled.
pub fn compile_kernel(source: &str, language: Language) -> Result<Vec<u8>, RuntimeError> {
    let config = CompilerConfig {
        language,
        opt_level: OptLevel::O2,
        ..CompilerConfig::default()
    };
    let wbin = wave_compiler::compile_source(source, &config)?;
    Ok(wbin)
}

/// Compile kernel source code with a custom configuration.
///
/// # Errors
///
/// Returns `RuntimeError::Compile` if the source cannot be parsed or compiled.
pub fn compile_kernel_with_config(
    source: &str,
    config: &CompilerConfig,
) -> Result<Vec<u8>, RuntimeError> {
    let wbin = wave_compiler::compile_source(source, config)?;
    Ok(wbin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_python_kernel() {
        let source = r#"
@kernel
def vector_add(a: Buffer[f32], b: Buffer[f32], out: Buffer[f32], n: u32):
    gid = thread_id()
    if gid < n:
        out[gid] = a[gid] + b[gid]
"#;
        let result = compile_kernel(source, Language::Python);
        assert!(result.is_ok());
        let wbin = result.unwrap();
        assert_eq!(&wbin[0..4], b"WAVE");
    }
}
