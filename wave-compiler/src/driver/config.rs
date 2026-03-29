// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Compiler configuration: optimization level, target parameters, and debug flags.
//!
//! Controls how the compiler behaves during compilation, including which
//! optimization passes to run and what target constraints to enforce.

/// Optimization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimizations.
    O0,
    /// Basic optimizations.
    O1,
    /// Standard optimizations.
    O2,
    /// Aggressive optimizations.
    O3,
}

impl OptLevel {
    /// Parse an optimization level from a string.
    #[must_use]
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "0" | "O0" => Some(Self::O0),
            "1" | "O1" => Some(Self::O1),
            "2" | "O2" => Some(Self::O2),
            "3" | "O3" => Some(Self::O3),
            _ => None,
        }
    }
}

/// Source language of the input file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    /// Python source.
    Python,
    /// Rust source.
    Rust,
    /// C/C++ source.
    Cpp,
    /// TypeScript source.
    TypeScript,
}

impl Language {
    /// Parse a language from a string.
    #[must_use]
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "python" | "py" => Some(Self::Python),
            "rust" | "rs" => Some(Self::Rust),
            "cpp" | "c++" | "c" => Some(Self::Cpp),
            "typescript" | "ts" | "javascript" | "js" => Some(Self::TypeScript),
            _ => None,
        }
    }
}

/// Compiler configuration.
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    /// Source language.
    pub language: Language,
    /// Optimization level.
    pub opt_level: OptLevel,
    /// Maximum registers for allocation.
    pub max_registers: u32,
    /// Wave width (SIMD lanes).
    pub wave_width: u32,
    /// Whether to dump HIR.
    pub dump_hir: bool,
    /// Whether to dump MIR.
    pub dump_mir: bool,
    /// Whether to dump LIR.
    pub dump_lir: bool,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            language: Language::Python,
            opt_level: OptLevel::O0,
            max_registers: 256,
            wave_width: 32,
            dump_hir: false,
            dump_mir: false,
            dump_lir: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opt_level_parse() {
        assert_eq!(OptLevel::from_str_opt("0"), Some(OptLevel::O0));
        assert_eq!(OptLevel::from_str_opt("O2"), Some(OptLevel::O2));
        assert_eq!(OptLevel::from_str_opt("3"), Some(OptLevel::O3));
        assert_eq!(OptLevel::from_str_opt("invalid"), None);
    }

    #[test]
    fn test_language_parse() {
        assert_eq!(Language::from_str_opt("python"), Some(Language::Python));
        assert_eq!(Language::from_str_opt("py"), Some(Language::Python));
        assert_eq!(Language::from_str_opt("rust"), Some(Language::Rust));
        assert_eq!(Language::from_str_opt("cpp"), Some(Language::Cpp));
        assert_eq!(Language::from_str_opt("ts"), Some(Language::TypeScript));
        assert_eq!(Language::from_str_opt("unknown"), None);
    }

    #[test]
    fn test_default_config() {
        let config = CompilerConfig::default();
        assert_eq!(config.language, Language::Python);
        assert_eq!(config.opt_level, OptLevel::O0);
        assert_eq!(config.max_registers, 256);
        assert_eq!(config.wave_width, 32);
    }
}
