// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Control flow graph construction and traversal utilities.
//!
//! Builds predecessor/successor maps and provides reverse postorder
//! traversal for dataflow analysis algorithms.

use std::collections::{HashMap, HashSet};

use crate::mir::function::MirFunction;
use crate::mir::value::BlockId;

/// Control flow graph with predecessor and successor maps.
pub struct Cfg {
    /// Predecessors for each block.
    pub predecessors: HashMap<BlockId, Vec<BlockId>>,
    /// Successors for each block.
    pub successors: HashMap<BlockId, Vec<BlockId>>,
    /// Entry block.
    pub entry: BlockId,
    /// All block IDs in the function.
    pub blocks: Vec<BlockId>,
}

impl Cfg {
    /// Build a CFG from a MIR function.
    #[must_use]
    pub fn build(func: &MirFunction) -> Self {
        let mut predecessors: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
        let mut successors: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
        let blocks: Vec<BlockId> = func.blocks.iter().map(|b| b.id).collect();

        for &bid in &blocks {
            predecessors.entry(bid).or_default();
            successors.entry(bid).or_default();
        }

        for block in &func.blocks {
            let succs = block.successors();
            successors.insert(block.id, succs.clone());
            for succ in succs {
                predecessors.entry(succ).or_default().push(block.id);
            }
        }

        Self {
            predecessors,
            successors,
            entry: func.entry,
            blocks,
        }
    }

    /// Compute reverse postorder traversal of the CFG.
    #[must_use]
    pub fn reverse_postorder(&self) -> Vec<BlockId> {
        let mut visited = HashSet::new();
        let mut postorder = Vec::new();
        self.dfs_postorder(self.entry, &mut visited, &mut postorder);
        postorder.reverse();
        postorder
    }

    fn dfs_postorder(
        &self,
        block: BlockId,
        visited: &mut HashSet<BlockId>,
        postorder: &mut Vec<BlockId>,
    ) {
        if !visited.insert(block) {
            return;
        }
        if let Some(succs) = self.successors.get(&block) {
            for succ in succs {
                self.dfs_postorder(*succ, visited, postorder);
            }
        }
        postorder.push(block);
    }

    /// Returns the predecessors of a block.
    #[must_use]
    pub fn preds(&self, block: BlockId) -> &[BlockId] {
        self.predecessors.get(&block).map_or(&[], |v| v.as_slice())
    }

    /// Returns the successors of a block.
    #[must_use]
    pub fn succs(&self, block: BlockId) -> &[BlockId] {
        self.successors.get(&block).map_or(&[], |v| v.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::basic_block::{BasicBlock, Terminator};
    use crate::mir::value::ValueId;

    fn make_diamond_cfg() -> MirFunction {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb0 = BasicBlock::new(BlockId(0));
        bb0.terminator = Terminator::CondBranch {
            cond: ValueId(0),
            true_target: BlockId(1),
            false_target: BlockId(2),
        };
        let mut bb1 = BasicBlock::new(BlockId(1));
        bb1.terminator = Terminator::Branch { target: BlockId(3) };
        let mut bb2 = BasicBlock::new(BlockId(2));
        bb2.terminator = Terminator::Branch { target: BlockId(3) };
        let bb3 = BasicBlock::new(BlockId(3));
        func.blocks.push(bb0);
        func.blocks.push(bb1);
        func.blocks.push(bb2);
        func.blocks.push(bb3);
        func
    }

    #[test]
    fn test_cfg_predecessors() {
        let func = make_diamond_cfg();
        let cfg = Cfg::build(&func);
        assert!(cfg.preds(BlockId(0)).is_empty());
        assert_eq!(cfg.preds(BlockId(1)), &[BlockId(0)]);
        assert_eq!(cfg.preds(BlockId(2)), &[BlockId(0)]);
        let preds3 = cfg.preds(BlockId(3));
        assert_eq!(preds3.len(), 2);
        assert!(preds3.contains(&BlockId(1)));
        assert!(preds3.contains(&BlockId(2)));
    }

    #[test]
    fn test_reverse_postorder() {
        let func = make_diamond_cfg();
        let cfg = Cfg::build(&func);
        let rpo = cfg.reverse_postorder();
        assert_eq!(rpo[0], BlockId(0));
        assert_eq!(*rpo.last().unwrap(), BlockId(3));
        assert_eq!(rpo.len(), 4);
    }

    #[test]
    fn test_cfg_successors() {
        let func = make_diamond_cfg();
        let cfg = Cfg::build(&func);
        assert_eq!(cfg.succs(BlockId(0)), &[BlockId(1), BlockId(2)]);
        assert_eq!(cfg.succs(BlockId(1)), &[BlockId(3)]);
        assert_eq!(cfg.succs(BlockId(3)), &[]);
    }
}
