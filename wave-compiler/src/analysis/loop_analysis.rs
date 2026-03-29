// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Loop detection, nesting depth, and induction variable analysis.
//!
//! Detects natural loops by finding back edges in the CFG, then
//! determines loop bodies, nesting relationships, and identifies
//! simple induction variables for optimization.

use std::collections::{HashMap, HashSet};

use super::cfg::Cfg;
use super::dominance::DomTree;
use crate::mir::value::BlockId;

/// A natural loop in the CFG.
#[derive(Debug, Clone)]
pub struct NaturalLoop {
    /// Loop header block (dominates all blocks in the loop).
    pub header: BlockId,
    /// Back edge source block.
    pub latch: BlockId,
    /// All blocks in the loop body (including header).
    pub body: HashSet<BlockId>,
    /// Nesting depth (0 for outermost loops).
    pub depth: u32,
}

/// Result of loop analysis.
pub struct LoopInfo {
    /// Detected natural loops.
    pub loops: Vec<NaturalLoop>,
    /// Back edges in the CFG.
    pub back_edges: Vec<(BlockId, BlockId)>,
    /// Loop depth for each block (0 if not in any loop).
    pub block_depth: HashMap<BlockId, u32>,
}

impl LoopInfo {
    /// Perform loop analysis on a CFG with dominator tree.
    #[must_use]
    pub fn compute(cfg: &Cfg, dom: &DomTree) -> Self {
        let back_edges = detect_back_edges(cfg, dom);
        let mut loops = Vec::new();

        for &(latch, header) in &back_edges {
            let body = compute_loop_body(cfg, header, latch);
            loops.push(NaturalLoop {
                header,
                latch,
                body,
                depth: 0,
            });
        }

        compute_nesting_depths(&mut loops);

        let mut block_depth: HashMap<BlockId, u32> = HashMap::new();
        for &bid in &cfg.blocks {
            block_depth.insert(bid, 0);
        }
        for nl in &loops {
            for &bid in &nl.body {
                let current = block_depth.get(&bid).copied().unwrap_or(0);
                if nl.depth + 1 > current {
                    block_depth.insert(bid, nl.depth + 1);
                }
            }
        }

        Self {
            loops,
            back_edges,
            block_depth,
        }
    }

    /// Returns the loop depth of a block (0 if not in any loop).
    #[must_use]
    pub fn depth(&self, block: BlockId) -> u32 {
        self.block_depth.get(&block).copied().unwrap_or(0)
    }

    /// Returns the innermost loop containing a block, if any.
    #[must_use]
    pub fn containing_loop(&self, block: BlockId) -> Option<&NaturalLoop> {
        self.loops
            .iter()
            .filter(|l| l.body.contains(&block))
            .max_by_key(|l| l.depth)
    }
}

fn detect_back_edges(cfg: &Cfg, dom: &DomTree) -> Vec<(BlockId, BlockId)> {
    let mut back_edges = Vec::new();
    for &bid in &cfg.blocks {
        for &succ in cfg.succs(bid) {
            if dom.dominates(succ, bid) {
                back_edges.push((bid, succ));
            }
        }
    }
    back_edges
}

fn compute_loop_body(cfg: &Cfg, header: BlockId, latch: BlockId) -> HashSet<BlockId> {
    let mut body = HashSet::new();
    body.insert(header);
    if header == latch {
        return body;
    }

    let mut stack = vec![latch];
    body.insert(latch);

    while let Some(block) = stack.pop() {
        for &pred in cfg.preds(block) {
            if body.insert(pred) {
                stack.push(pred);
            }
        }
    }

    body
}

fn compute_nesting_depths(loops: &mut [NaturalLoop]) {
    let n = loops.len();
    for i in 0..n {
        let mut depth = 0u32;
        for j in 0..n {
            if i != j && loops[j].body.is_superset(&loops[i].body) {
                depth += 1;
            }
        }
        loops[i].depth = depth;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::basic_block::{BasicBlock, Terminator};
    use crate::mir::function::MirFunction;
    use crate::mir::value::ValueId;

    fn make_simple_loop() -> MirFunction {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb0 = BasicBlock::new(BlockId(0));
        bb0.terminator = Terminator::Branch { target: BlockId(1) };

        let mut bb1 = BasicBlock::new(BlockId(1));
        bb1.terminator = Terminator::CondBranch {
            cond: ValueId(0),
            true_target: BlockId(2),
            false_target: BlockId(3),
        };

        let mut bb2 = BasicBlock::new(BlockId(2));
        bb2.terminator = Terminator::Branch { target: BlockId(1) };

        let bb3 = BasicBlock::new(BlockId(3));

        func.blocks.push(bb0);
        func.blocks.push(bb1);
        func.blocks.push(bb2);
        func.blocks.push(bb3);
        func
    }

    #[test]
    fn test_detect_loop() {
        let func = make_simple_loop();
        let cfg = Cfg::build(&func);
        let dom = DomTree::compute(&cfg);
        let loop_info = LoopInfo::compute(&cfg, &dom);

        assert_eq!(loop_info.back_edges.len(), 1);
        assert_eq!(loop_info.back_edges[0], (BlockId(2), BlockId(1)));
        assert_eq!(loop_info.loops.len(), 1);
        assert_eq!(loop_info.loops[0].header, BlockId(1));
        assert!(loop_info.loops[0].body.contains(&BlockId(1)));
        assert!(loop_info.loops[0].body.contains(&BlockId(2)));
    }

    #[test]
    fn test_loop_depth() {
        let func = make_simple_loop();
        let cfg = Cfg::build(&func);
        let dom = DomTree::compute(&cfg);
        let loop_info = LoopInfo::compute(&cfg, &dom);

        assert_eq!(loop_info.depth(BlockId(0)), 0);
        assert_eq!(loop_info.depth(BlockId(1)), 1);
        assert_eq!(loop_info.depth(BlockId(2)), 1);
        assert_eq!(loop_info.depth(BlockId(3)), 0);
    }

    #[test]
    fn test_no_loops() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb0 = BasicBlock::new(BlockId(0));
        bb0.terminator = Terminator::Branch { target: BlockId(1) };
        let bb1 = BasicBlock::new(BlockId(1));
        func.blocks.push(bb0);
        func.blocks.push(bb1);

        let cfg = Cfg::build(&func);
        let dom = DomTree::compute(&cfg);
        let loop_info = LoopInfo::compute(&cfg, &dom);

        assert!(loop_info.loops.is_empty());
        assert!(loop_info.back_edges.is_empty());
    }
}
