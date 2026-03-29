// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Compilation session state tracking.
//!
//! Maintains state across the compilation pipeline including
//! configuration, source information, and compilation statistics.

use super::config::CompilerConfig;
use crate::diagnostics::SourceMap;

/// Compilation session holding all state for a single compilation.
pub struct Session {
    /// Compiler configuration.
    pub config: CompilerConfig,
    /// Source map for error reporting.
    pub source_map: Option<SourceMap>,
    /// Input file path.
    pub input_path: String,
    /// Output file path.
    pub output_path: String,
}

impl Session {
    /// Create a new compilation session.
    #[must_use]
    pub fn new(config: CompilerConfig, input_path: String, output_path: String) -> Self {
        Self {
            config,
            source_map: None,
            input_path,
            output_path,
        }
    }

    /// Set the source code for error reporting.
    pub fn set_source(&mut self, source: String) {
        self.source_map = Some(SourceMap::new(source));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation() {
        let config = CompilerConfig::default();
        let session = Session::new(config, "input.py".into(), "output.wbin".into());
        assert_eq!(session.input_path, "input.py");
        assert_eq!(session.output_path, "output.wbin");
        assert!(session.source_map.is_none());
    }

    #[test]
    fn test_session_set_source() {
        let config = CompilerConfig::default();
        let mut session = Session::new(config, "input.py".into(), "output.wbin".into());
        session.set_source("gid = thread_id()".into());
        assert!(session.source_map.is_some());
    }
}
