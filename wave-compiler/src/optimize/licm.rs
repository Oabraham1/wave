// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Loop-invariant code motion pass.
//!
//! Identifies instructions inside loops whose operands do not change
//! across iterations and moves them to the loop preheader.

use std::collections::HashSet;

use super::pass::Pass;
use crate::analysis::cfg::Cfg;
use crate::analysis::dominance::DomTree;
use crate::analysis::loop_analysis::LoopInfo;
use crate::mir::function::MirFunction;
use crate::mir::instruction::MirInst;
use crate::mir::value::ValueId;

/// Loop-invariant code motion pass.
pub struct Licm;

impl Pass for Licm {
    fn name(&self) -> &str {
        "licm"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let cfg = Cfg::build(func);
        let dom = DomTree::compute(&cfg);
        let loop_info = LoopInfo::compute(&cfg, &dom);

        let mut changed = false;

        for natural_loop in &loop_info.loops {
            let mut defs_in_loop: HashSet<ValueId> = HashSet::new();
            for &bid in &natural_loop.body {
                if let Some(block) = func.block(bid) {
                    for inst in &block.instructions {
                        if let Some(dest) = inst.dest() {
                            defs_in_loop.insert(dest);
                        }
                    }
                }
            }

            let mut invariant_insts: Vec<(usize, MirInst)> = Vec::new();

            for &bid in &natural_loop.body {
                if let Some(block) = func.block(bid) {
                    for (idx, inst) in block.instructions.iter().enumerate() {
                        if inst.has_side_effects() {
                            continue;
                        }
                        let all_operands_invariant = inst
                            .operands()
                            .iter()
                            .all(|op| !defs_in_loop.contains(op));
                        if all_operands_invariant {
                            if let Some(dest) = inst.dest() {
                                invariant_insts.push((idx, inst.clone()));
                                defs_in_loop.remove(&dest);
                            }
                        }
                    }
                }
            }

            if !invariant_insts.is_empty() {
                let preds = cfg.preds(natural_loop.header);
                let preheader = preds
                    .iter()
                    .find(|p| !natural_loop.body.contains(p))
                    .copied();

                if let Some(pre_bid) = preheader {
                    if let Some(pre_block) = func.block_mut(pre_bid) {
                        for (_, inst) in &invariant_insts {
                            pre_block.instructions.push(inst.clone());
                        }
                        changed = true;
                    }
                }
            }
        }

        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::expr::BinOp;
    use crate::mir::basic_block::{BasicBlock, Terminator};
    use crate::mir::instruction::{ConstValue, MirInst};
    use crate::mir::types::MirType;
    use crate::mir::value::BlockId;

    #[test]
    fn test_licm_no_loops_no_change() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb = BasicBlock::new(BlockId(0));
        bb.instructions.push(MirInst::Const {
            dest: ValueId(0),
            value: ConstValue::I32(42),
        });
        bb.terminator = Terminator::Return;
        func.blocks.push(bb);

        let pass = Licm;
        assert!(!pass.run(&mut func));
    }

    #[test]
    fn test_licm_hoists_invariant() {
        let mut func = MirFunction::new("test".into(), BlockId(0));

        let mut bb0 = BasicBlock::new(BlockId(0));
        bb0.terminator = Terminator::Branch { target: BlockId(1) };

        let mut bb1 = BasicBlock::new(BlockId(1));
        bb1.instructions.push(MirInst::BinOp {
            dest: ValueId(2),
            op: BinOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: MirType::I32,
        });
        bb1.terminator = Terminator::CondBranch {
            cond: ValueId(2),
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

        let pass = Licm;
        let changed = pass.run(&mut func);
        assert!(changed);
        assert!(!func.blocks[0].instructions.is_empty());
    }
}
