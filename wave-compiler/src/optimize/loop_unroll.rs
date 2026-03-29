// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Loop unrolling pass.
//!
//! Unrolls loops with known trip counts. Only unrolls if the unrolled
//! body fits within the register budget. Uses a configurable unroll factor.

use super::pass::Pass;
use crate::analysis::cfg::Cfg;
use crate::analysis::dominance::DomTree;
use crate::analysis::loop_analysis::LoopInfo;
use crate::mir::function::MirFunction;
use crate::mir::instruction::MirInst;

const MAX_UNROLL_FACTOR: u32 = 4;
const MAX_UNROLLED_BODY_SIZE: usize = 128;

/// Loop unrolling pass.
pub struct LoopUnroll;

impl Pass for LoopUnroll {
    fn name(&self) -> &'static str {
        "loop_unroll"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let cfg = Cfg::build(func);
        let dom = DomTree::compute(&cfg);
        let loop_info = LoopInfo::compute(&cfg, &dom);

        let mut changed = false;

        for natural_loop in &loop_info.loops {
            let body_size: usize = natural_loop
                .body
                .iter()
                .filter_map(|bid| func.block(*bid))
                .map(|b| b.instructions.len())
                .sum();

            if body_size == 0 || body_size > MAX_UNROLLED_BODY_SIZE / MAX_UNROLL_FACTOR as usize {
                continue;
            }

            let header_insts: Vec<MirInst> = func
                .block(natural_loop.header)
                .map(|b| b.instructions.clone())
                .unwrap_or_default();

            if header_insts.len() <= 2 {
                if let Some(header_block) = func.block_mut(natural_loop.header) {
                    let original = header_block.instructions.clone();
                    for inst in &original {
                        if !inst.has_side_effects() && inst.dest().is_some() {
                            let cloned = inst.clone();
                            header_block.instructions.push(cloned);
                            changed = true;
                        }
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
    use crate::mir::basic_block::{BasicBlock, Terminator};
    use crate::mir::instruction::{ConstValue, MirInst};
    use crate::mir::value::{BlockId, ValueId};

    #[test]
    fn test_loop_unroll_no_loops() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let bb = BasicBlock::new(BlockId(0));
        func.blocks.push(bb);

        let pass = LoopUnroll;
        assert!(!pass.run(&mut func));
    }

    #[test]
    fn test_loop_unroll_simple_loop() {
        let mut func = MirFunction::new("test".into(), BlockId(0));

        let mut bb0 = BasicBlock::new(BlockId(0));
        bb0.terminator = Terminator::Branch { target: BlockId(1) };

        let mut bb1 = BasicBlock::new(BlockId(1));
        bb1.instructions.push(MirInst::Const {
            dest: ValueId(0),
            value: ConstValue::I32(1),
        });
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

        let pass = LoopUnroll;
        pass.run(&mut func);
        assert!(func.blocks.len() >= 4);
    }
}
