// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! SSA value identifiers for MIR.
//!
//! Values in MIR are in Static Single Assignment form. Each value is
//! assigned exactly once and identified by a unique `ValueId`.

/// Unique identifier for an SSA value (virtual register).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub u32);

impl ValueId {
    /// Returns the raw numeric ID.
    #[must_use]
    pub fn index(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for ValueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// Unique identifier for a basic block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

impl BlockId {
    /// Returns the raw numeric ID.
    #[must_use]
    pub fn index(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

/// Generator for unique value and block IDs.
pub struct IdGenerator {
    next_value: u32,
    next_block: u32,
}

impl IdGenerator {
    /// Create a new ID generator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_value: 0,
            next_block: 0,
        }
    }

    /// Generate a fresh value ID.
    pub fn next_value(&mut self) -> ValueId {
        let id = ValueId(self.next_value);
        self.next_value += 1;
        id
    }

    /// Generate a fresh block ID.
    pub fn next_block(&mut self) -> BlockId {
        let id = BlockId(self.next_block);
        self.next_block += 1;
        id
    }

    /// Returns the count of values generated so far.
    #[must_use]
    pub fn value_count(&self) -> u32 {
        self.next_value
    }
}

impl Default for IdGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_generator_values() {
        let mut gen = IdGenerator::new();
        let v0 = gen.next_value();
        let v1 = gen.next_value();
        assert_eq!(v0, ValueId(0));
        assert_eq!(v1, ValueId(1));
        assert_eq!(gen.value_count(), 2);
    }

    #[test]
    fn test_id_generator_blocks() {
        let mut gen = IdGenerator::new();
        let b0 = gen.next_block();
        let b1 = gen.next_block();
        assert_eq!(b0, BlockId(0));
        assert_eq!(b1, BlockId(1));
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", ValueId(5)), "%5");
        assert_eq!(format!("{}", BlockId(3)), "bb3");
    }
}
