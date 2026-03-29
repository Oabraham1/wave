// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Source location tracking for error messages.
//!
//! Maps byte offsets in compiled code back to source file positions
//! for user-facing error reporting.

use super::error::SourceLoc;

/// Maps source code positions to line/column numbers.
pub struct SourceMap {
    source: String,
    line_starts: Vec<usize>,
}

impl SourceMap {
    /// Create a new source map from source code.
    #[must_use]
    pub fn new(source: String) -> Self {
        let mut line_starts = vec![0];
        for (i, ch) in source.char_indices() {
            if ch == '\n' {
                line_starts.push(i + 1);
            }
        }
        Self {
            source,
            line_starts,
        }
    }

    /// Convert a byte offset to a source location.
    #[must_use]
    pub fn offset_to_loc(&self, offset: usize) -> SourceLoc {
        let line = self
            .line_starts
            .partition_point(|&start| start <= offset)
            .saturating_sub(1);
        let col = offset - self.line_starts[line];
        SourceLoc {
            #[allow(clippy::cast_possible_truncation)]
            line: (line + 1) as u32,
            #[allow(clippy::cast_possible_truncation)]
            col: (col + 1) as u32,
        }
    }

    /// Get a line of source code by line number (1-based).
    #[must_use]
    pub fn get_line(&self, line: u32) -> Option<&str> {
        let idx = (line as usize).checked_sub(1)?;
        let start = *self.line_starts.get(idx)?;
        let end = self
            .line_starts
            .get(idx + 1)
            .map_or(self.source.len(), |&s| s.saturating_sub(1));
        Some(&self.source[start..end])
    }

    /// Returns the total number of lines.
    #[must_use]
    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offset_to_loc() {
        let src = "line1\nline2\nline3".to_string();
        let map = SourceMap::new(src);
        assert_eq!(map.offset_to_loc(0), SourceLoc { line: 1, col: 1 });
        assert_eq!(map.offset_to_loc(6), SourceLoc { line: 2, col: 1 });
        assert_eq!(map.offset_to_loc(8), SourceLoc { line: 2, col: 3 });
    }

    #[test]
    fn test_get_line() {
        let src = "first\nsecond\nthird".to_string();
        let map = SourceMap::new(src);
        assert_eq!(map.get_line(1), Some("first"));
        assert_eq!(map.get_line(2), Some("second"));
        assert_eq!(map.get_line(3), Some("third"));
        assert_eq!(map.get_line(4), None);
    }

    #[test]
    fn test_line_count() {
        let src = "a\nb\nc\n".to_string();
        let map = SourceMap::new(src);
        assert_eq!(map.line_count(), 4);
    }
}
