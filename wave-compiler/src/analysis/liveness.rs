// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Compute live-in and live-out values for each block using backward dataflow analysis.

use std::collections::{HashMap, HashSet};

use super::cfg::Cfg;
use crate::mir::function::MirFunction;
use crate::mir::value::{BlockId, ValueId};

#[derive(Debug)]
/// Result of live variable analysis.
pub struct LivenessInfo {
    /// Values live on entry to each block.
    pub live_in: HashMap<BlockId, HashSet<ValueId>>,
    /// Values live on exit from each block.
    pub live_out: HashMap<BlockId, HashSet<ValueId>>,
}

impl LivenessInfo {
    /// Compute liveness information for a MIR function.
    #[must_use]
    pub fn compute(func: &MirFunction, cfg: &Cfg) -> Self {
        let mut live_in: HashMap<BlockId, HashSet<ValueId>> = HashMap::new();
        let mut live_out: HashMap<BlockId, HashSet<ValueId>> = HashMap::new();

        for &bid in &cfg.blocks {
            live_in.insert(bid, HashSet::new());
            live_out.insert(bid, HashSet::new());
        }

        let mut use_sets: HashMap<BlockId, HashSet<ValueId>> = HashMap::new();
        let mut def_sets: HashMap<BlockId, HashSet<ValueId>> = HashMap::new();

        for block in &func.blocks {
            let mut uses = HashSet::new();
            let mut defs = HashSet::new();

            for phi in &block.phis {
                defs.insert(phi.dest);
            }

            for inst in &block.instructions {
                for operand in inst.operand_iter() {
                    if !defs.contains(&operand) {
                        uses.insert(operand);
                    }
                }
                if let Some(dest) = inst.dest() {
                    defs.insert(dest);
                }
            }

            for operand in block.terminator.operand_iter() {
                if !defs.contains(&operand) {
                    uses.insert(operand);
                }
            }

            use_sets.insert(block.id, uses);
            def_sets.insert(block.id, defs);
        }

        let mut edge_uses: HashMap<(BlockId, BlockId), HashSet<ValueId>> = HashMap::new();
        for block in &func.blocks {
            for phi in &block.phis {
                for &(predecessor, value) in &phi.incoming {
                    edge_uses
                        .entry((predecessor, block.id))
                        .or_default()
                        .insert(value);
                }
            }
        }

        let mut changed = true;
        while changed {
            changed = false;
            for &bid in cfg.blocks.iter().rev() {
                let mut new_out: HashSet<ValueId> = HashSet::new();
                for succ in cfg.succs(bid) {
                    if let Some(values) = edge_uses.get(&(bid, *succ)) {
                        new_out.extend(values);
                    }
                    if let Some(live) = live_in.get(succ) {
                        new_out.extend(live);
                    }
                }

                let mut new_in = use_sets.get(&bid).cloned().unwrap_or_default();
                let defs = def_sets.get(&bid);
                new_in.extend(
                    new_out
                        .iter()
                        .filter(|value| !defs.is_some_and(|defined| defined.contains(value)))
                        .copied(),
                );

                if new_in != *live_in.get(&bid).unwrap_or(&HashSet::new()) {
                    live_in.insert(bid, new_in);
                    changed = true;
                }
                if new_out != *live_out.get(&bid).unwrap_or(&HashSet::new()) {
                    live_out.insert(bid, new_out);
                    changed = true;
                }
            }
        }

        Self { live_in, live_out }
    }

    /// Check if a value is live at the entry of a block.
    #[must_use]
    pub fn is_live_in(&self, block: BlockId, value: ValueId) -> bool {
        self.live_in.get(&block).is_some_and(|s| s.contains(&value))
    }

    /// Check if a value is live at the exit of a block.
    #[must_use]
    pub fn is_live_out(&self, block: BlockId, value: ValueId) -> bool {
        self.live_out
            .get(&block)
            .is_some_and(|s| s.contains(&value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::expr::BinOp;
    use crate::mir::basic_block::{BasicBlock, Terminator};
    use crate::mir::instruction::{ConstValue, MirInst};
    use crate::mir::types::MirType;

    #[test]
    fn test_liveness_simple() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb0 = BasicBlock::new(BlockId(0));
        bb0.instructions.push(MirInst::Const {
            dest: ValueId(0),
            value: ConstValue::I32(1),
        });
        bb0.instructions.push(MirInst::Const {
            dest: ValueId(1),
            value: ConstValue::I32(2),
        });
        bb0.terminator = Terminator::Branch { target: BlockId(1) };

        let mut bb1 = BasicBlock::new(BlockId(1));
        bb1.instructions.push(MirInst::BinOp {
            dest: ValueId(2),
            op: BinOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: MirType::I32,
        });
        bb1.terminator = Terminator::Return;

        func.blocks.push(bb0);
        func.blocks.push(bb1);

        let cfg = Cfg::build(&func);
        let liveness = LivenessInfo::compute(&func, &cfg);

        assert!(liveness.is_live_out(BlockId(0), ValueId(0)));
        assert!(liveness.is_live_out(BlockId(0), ValueId(1)));
        assert!(liveness.is_live_in(BlockId(1), ValueId(0)));
        assert!(liveness.is_live_in(BlockId(1), ValueId(1)));
    }

    #[test]
    fn test_liveness_dead_value() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb0 = BasicBlock::new(BlockId(0));
        bb0.instructions.push(MirInst::Const {
            dest: ValueId(0),
            value: ConstValue::I32(1),
        });
        bb0.instructions.push(MirInst::Const {
            dest: ValueId(1),
            value: ConstValue::I32(2),
        });
        bb0.terminator = Terminator::Return;
        func.blocks.push(bb0);

        let cfg = Cfg::build(&func);
        let liveness = LivenessInfo::compute(&func, &cfg);

        assert!(!liveness.is_live_out(BlockId(0), ValueId(0)));
        assert!(!liveness.is_live_out(BlockId(0), ValueId(1)));
    }
}
