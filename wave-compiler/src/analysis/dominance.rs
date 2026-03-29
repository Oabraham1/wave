// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Dominator tree and dominance frontier computation.
//!
//! Uses the Cooper-Harvey-Kennedy algorithm for efficient dominator tree
//! construction. The dominance frontier is used by SSA construction (mem2reg).

use std::collections::{HashMap, HashSet};

use super::cfg::Cfg;
use crate::mir::value::BlockId;

/// Dominator tree for a CFG.
pub struct DomTree {
    /// Immediate dominator of each block (entry has no idom).
    pub idom: HashMap<BlockId, BlockId>,
    /// Children in the dominator tree.
    pub children: HashMap<BlockId, Vec<BlockId>>,
    /// Dominance frontier for each block.
    pub frontiers: HashMap<BlockId, HashSet<BlockId>>,
}

impl DomTree {
    /// Compute the dominator tree using the Cooper-Harvey-Kennedy algorithm.
    #[must_use]
    pub fn compute(cfg: &Cfg) -> Self {
        let rpo = cfg.reverse_postorder();
        let mut block_to_rpo: HashMap<BlockId, usize> = HashMap::new();
        for (i, &bid) in rpo.iter().enumerate() {
            block_to_rpo.insert(bid, i);
        }

        let mut idom: HashMap<BlockId, BlockId> = HashMap::new();
        idom.insert(cfg.entry, cfg.entry);

        let mut changed = true;
        while changed {
            changed = false;
            for &bid in &rpo {
                if bid == cfg.entry {
                    continue;
                }
                let preds = cfg.preds(bid);
                let mut new_idom: Option<BlockId> = None;

                for &pred in preds {
                    if idom.contains_key(&pred) {
                        new_idom = Some(match new_idom {
                            None => pred,
                            Some(current) => intersect(current, pred, &idom, &block_to_rpo),
                        });
                    }
                }

                if let Some(new_id) = new_idom {
                    if idom.get(&bid) != Some(&new_id) {
                        idom.insert(bid, new_id);
                        changed = true;
                    }
                }
            }
        }

        let mut children: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
        for &bid in &cfg.blocks {
            children.entry(bid).or_default();
        }
        for (&bid, &dom) in &idom {
            if bid != dom {
                children.entry(dom).or_default().push(bid);
            }
        }

        let frontiers = compute_frontiers(cfg, &idom);

        Self {
            idom,
            children,
            frontiers,
        }
    }

    /// Returns true if `a` dominates `b`.
    #[must_use]
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        if a == b {
            return true;
        }
        let mut current = b;
        loop {
            match self.idom.get(&current) {
                Some(&dom) if dom == current => return false,
                Some(&dom) if dom == a => return true,
                Some(&dom) => current = dom,
                None => return false,
            }
        }
    }

    /// Returns the dominance frontier of a block.
    #[must_use]
    pub fn frontier(&self, block: BlockId) -> HashSet<BlockId> {
        self.frontiers.get(&block).cloned().unwrap_or_default()
    }
}

fn intersect(
    mut a: BlockId,
    mut b: BlockId,
    idom: &HashMap<BlockId, BlockId>,
    rpo: &HashMap<BlockId, usize>,
) -> BlockId {
    while a != b {
        let a_rpo = rpo.get(&a).copied().unwrap_or(usize::MAX);
        let b_rpo = rpo.get(&b).copied().unwrap_or(usize::MAX);
        if a_rpo > b_rpo {
            a = *idom.get(&a).unwrap_or(&a);
        } else {
            b = *idom.get(&b).unwrap_or(&b);
        }
    }
    a
}

fn compute_frontiers(
    cfg: &Cfg,
    idom: &HashMap<BlockId, BlockId>,
) -> HashMap<BlockId, HashSet<BlockId>> {
    let mut frontiers: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();
    for &bid in &cfg.blocks {
        frontiers.entry(bid).or_default();
    }

    for &bid in &cfg.blocks {
        let preds = cfg.preds(bid);
        if preds.len() >= 2 {
            for &pred in preds {
                let mut runner = pred;
                while runner != *idom.get(&bid).unwrap_or(&bid) {
                    frontiers.entry(runner).or_default().insert(bid);
                    if runner == *idom.get(&runner).unwrap_or(&runner) {
                        break;
                    }
                    runner = *idom.get(&runner).unwrap_or(&runner);
                }
            }
        }
    }

    frontiers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::basic_block::{BasicBlock, Terminator};
    use crate::mir::function::MirFunction;
    use crate::mir::value::ValueId;

    fn make_diamond() -> MirFunction {
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
    fn test_dominator_tree() {
        let func = make_diamond();
        let cfg = Cfg::build(&func);
        let dom = DomTree::compute(&cfg);

        assert!(dom.dominates(BlockId(0), BlockId(1)));
        assert!(dom.dominates(BlockId(0), BlockId(2)));
        assert!(dom.dominates(BlockId(0), BlockId(3)));
        assert!(!dom.dominates(BlockId(1), BlockId(2)));
    }

    #[test]
    fn test_dominance_frontier() {
        let func = make_diamond();
        let cfg = Cfg::build(&func);
        let dom = DomTree::compute(&cfg);

        let df1 = dom.frontier(BlockId(1));
        assert!(df1.contains(&BlockId(3)));
        let df2 = dom.frontier(BlockId(2));
        assert!(df2.contains(&BlockId(3)));
        let df0 = dom.frontier(BlockId(0));
        assert!(df0.is_empty());
    }

    #[test]
    fn test_self_dominance() {
        let func = make_diamond();
        let cfg = Cfg::build(&func);
        let dom = DomTree::compute(&cfg);
        for bid in [BlockId(0), BlockId(1), BlockId(2), BlockId(3)] {
            assert!(dom.dominates(bid, bid));
        }
    }
}
