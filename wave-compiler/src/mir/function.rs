// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! MIR function/kernel representation with control flow graph.
//!
//! A `MirFunction` contains basic blocks forming a CFG, along with
//! type information for all SSA values and kernel parameter metadata.

use std::collections::HashMap;

use super::basic_block::BasicBlock;
use super::types::MirType;
use super::value::{BlockId, ValueId};

/// A kernel parameter in MIR form.
#[derive(Debug, Clone, PartialEq)]
pub struct MirParam {
    /// SSA value representing this parameter.
    pub value: ValueId,
    /// Parameter type.
    pub ty: MirType,
    /// Parameter name (for debugging).
    pub name: String,
}

/// A function/kernel in MIR with a control flow graph.
#[derive(Debug, Clone)]
pub struct MirFunction {
    /// Kernel name.
    pub name: String,
    /// Kernel parameters.
    pub params: Vec<MirParam>,
    /// Basic blocks forming the CFG.
    pub blocks: Vec<BasicBlock>,
    /// Entry block ID.
    pub entry: BlockId,
    /// Type mapping for all SSA values.
    pub value_types: HashMap<ValueId, MirType>,
}

impl MirFunction {
    /// Create a new MIR function.
    #[must_use]
    pub fn new(name: String, entry: BlockId) -> Self {
        Self {
            name,
            params: Vec::new(),
            blocks: Vec::new(),
            entry,
            value_types: HashMap::new(),
        }
    }

    /// Get a basic block by ID.
    #[must_use]
    pub fn block(&self, id: BlockId) -> Option<&BasicBlock> {
        self.blocks.iter().find(|b| b.id == id)
    }

    /// Get a mutable reference to a basic block by ID.
    pub fn block_mut(&mut self, id: BlockId) -> Option<&mut BasicBlock> {
        self.blocks.iter_mut().find(|b| b.id == id)
    }

    /// Returns all block IDs in the function.
    #[must_use]
    pub fn block_ids(&self) -> Vec<BlockId> {
        self.blocks.iter().map(|b| b.id).collect()
    }

    /// Returns the number of basic blocks.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Add a type mapping for a value.
    pub fn set_type(&mut self, value: ValueId, ty: MirType) {
        self.value_types.insert(value, ty);
    }

    /// Get the type of a value.
    #[must_use]
    pub fn get_type(&self, value: ValueId) -> Option<MirType> {
        self.value_types.get(&value).copied()
    }

    /// Compute predecessor blocks for each block.
    #[must_use]
    pub fn predecessors(&self) -> HashMap<BlockId, Vec<BlockId>> {
        let mut preds: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
        for block in &self.blocks {
            preds.entry(block.id).or_default();
            for succ in block.successors() {
                preds.entry(succ).or_default().push(block.id);
            }
        }
        preds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::basic_block::{BasicBlock, Terminator};

    #[test]
    fn test_mir_function_blocks() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb0 = BasicBlock::new(BlockId(0));
        bb0.terminator = Terminator::Branch {
            target: BlockId(1),
        };
        let bb1 = BasicBlock::new(BlockId(1));
        func.blocks.push(bb0);
        func.blocks.push(bb1);

        assert_eq!(func.block_count(), 2);
        assert!(func.block(BlockId(0)).is_some());
        assert!(func.block(BlockId(2)).is_none());
        assert_eq!(func.block_ids(), vec![BlockId(0), BlockId(1)]);
    }

    #[test]
    fn test_predecessors() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb0 = BasicBlock::new(BlockId(0));
        bb0.terminator = Terminator::CondBranch {
            cond: ValueId(0),
            true_target: BlockId(1),
            false_target: BlockId(2),
        };
        let bb1 = BasicBlock::new(BlockId(1));
        let bb2 = BasicBlock::new(BlockId(2));
        func.blocks.push(bb0);
        func.blocks.push(bb1);
        func.blocks.push(bb2);

        let preds = func.predecessors();
        assert!(preds[&BlockId(0)].is_empty());
        assert_eq!(preds[&BlockId(1)], vec![BlockId(0)]);
        assert_eq!(preds[&BlockId(2)], vec![BlockId(0)]);
    }

    #[test]
    fn test_value_types() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        func.set_type(ValueId(0), MirType::I32);
        func.set_type(ValueId(1), MirType::F32);
        assert_eq!(func.get_type(ValueId(0)), Some(MirType::I32));
        assert_eq!(func.get_type(ValueId(1)), Some(MirType::F32));
        assert_eq!(func.get_type(ValueId(99)), None);
    }
}
