// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Loop-invariant code motion pass. Identifies instructions inside loops whose operands do not
//! change across iterations and moves them to the loop preheader.

use std::collections::HashSet;

use super::pass::Pass;
use crate::analysis::cfg::Cfg;
use crate::analysis::dominance::DomTree;
use crate::analysis::loop_analysis::LoopInfo;
use crate::mir::function::MirFunction;
use crate::mir::instruction::MirInst;
use crate::mir::value::ValueId;

#[derive(Debug)]
/// Loop-invariant code motion pass.
pub struct Licm;

impl Pass for Licm {
    fn name(&self) -> &'static str {
        "licm"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let cfg = Cfg::build(func);
        let dom = DomTree::compute(&cfg);
        let loop_info = LoopInfo::compute(&cfg, &dom);

        let mut changed = false;

        for natural_loop in &loop_info.loops {
            let mut outside = cfg
                .preds(natural_loop.header)
                .iter()
                .filter(|bid| !natural_loop.body.contains(bid));
            let Some(&preheader) = outside.next() else {
                continue;
            };
            if outside.next().is_some() || cfg.succs(preheader) != [natural_loop.header] {
                continue;
            }

            let mut defs_in_loop: HashSet<ValueId> = func
                .blocks
                .iter()
                .filter(|block| natural_loop.body.contains(&block.id))
                .flat_map(|block| {
                    block
                        .phis
                        .iter()
                        .map(|phi| phi.dest)
                        .chain(block.instructions.iter().filter_map(MirInst::dest))
                })
                .collect();
            let mut hoisted = Vec::new();
            loop {
                let count = hoisted.len();
                for block in &mut func.blocks {
                    if !natural_loop.body.contains(&block.id) {
                        continue;
                    }
                    hoisted.extend(block.instructions.extract_if(.., |inst| {
                        let invariant = safe_to_hoist(inst)
                            && inst
                                .operand_iter()
                                .all(|value| !defs_in_loop.contains(&value));
                        if invariant {
                            if let Some(dest) = inst.dest() {
                                defs_in_loop.remove(&dest);
                            }
                        }
                        invariant
                    }));
                }
                if count == hoisted.len() {
                    break;
                }
            }
            if !hoisted.is_empty() {
                // CFG predecessors refer to blocks in this function.
                func.block_mut(preheader)
                    .expect("preheader belongs to the function")
                    .instructions
                    .extend(hoisted);
                changed = true;
            }
        }

        changed
    }
}

fn safe_to_hoist(inst: &MirInst) -> bool {
    use crate::hir::expr::BinOp;
    match inst {
        MirInst::BinOp { op, .. } => !matches!(op, BinOp::Div | BinOp::FloorDiv | BinOp::Mod),
        MirInst::Const { .. } | MirInst::UnaryOp { .. } | MirInst::Cast { .. } => true,
        MirInst::Load { .. }
        | MirInst::Store { .. }
        | MirInst::Call { .. }
        | MirInst::Shuffle { .. }
        | MirInst::ReadSpecialReg { .. }
        | MirInst::AtomicRmw { .. }
        | MirInst::Barrier
        | MirInst::Fence { .. } => false,
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
        assert_eq!(func.blocks[0].instructions.len(), 1);
        assert_eq!(func.blocks[1].instructions, [] as [MirInst; 0]);
        assert!(!pass.run(&mut func));
    }
}
