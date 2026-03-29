// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! CFG simplification pass.
//!
//! Merges basic blocks connected by unconditional branches when the
//! target has a single predecessor. Removes empty blocks and unreachable
//! blocks to simplify the control flow graph.

use std::collections::HashSet;

use super::pass::Pass;
use crate::mir::basic_block::Terminator;
use crate::mir::function::MirFunction;

/// CFG simplification pass.
pub struct SimplifyCfg;

impl Pass for SimplifyCfg {
    fn name(&self) -> &str {
        "simplify_cfg"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let mut changed = false;
        changed |= remove_unreachable_blocks(func);
        changed |= merge_single_predecessor_blocks(func);
        changed
    }
}

fn remove_unreachable_blocks(func: &mut MirFunction) -> bool {
    let mut reachable = HashSet::new();
    let mut stack = vec![func.entry];
    while let Some(bid) = stack.pop() {
        if !reachable.insert(bid) {
            continue;
        }
        if let Some(block) = func.block(bid) {
            for succ in block.successors() {
                stack.push(succ);
            }
        }
    }

    let original_count = func.blocks.len();
    func.blocks.retain(|b| reachable.contains(&b.id));
    func.blocks.len() != original_count
}

fn merge_single_predecessor_blocks(func: &mut MirFunction) -> bool {
    let preds = func.predecessors();
    let mut changed = false;

    loop {
        let mut merge_found = false;
        for i in 0..func.blocks.len() {
            let term = func.blocks[i].terminator.clone();
            if let Terminator::Branch { target } = term {
                if let Some(pred_list) = preds.get(&target) {
                    if pred_list.len() == 1 && pred_list[0] == func.blocks[i].id {
                        if let Some(target_idx) = func.blocks.iter().position(|b| b.id == target) {
                            if target_idx != i {
                                let target_block = func.blocks.remove(target_idx);
                                let adjusted_i = if target_idx < i { i - 1 } else { i };
                                func.blocks[adjusted_i]
                                    .instructions
                                    .extend(target_block.instructions);
                                func.blocks[adjusted_i].terminator = target_block.terminator;
                                merge_found = true;
                                changed = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
        if !merge_found {
            break;
        }
    }

    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::basic_block::BasicBlock;
    use crate::mir::instruction::{ConstValue, MirInst};
    use crate::mir::value::BlockId;
    use crate::mir::value::ValueId;

    #[test]
    fn test_remove_unreachable() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let bb0 = BasicBlock::new(BlockId(0));
        let bb1 = BasicBlock::new(BlockId(1));
        func.blocks.push(bb0);
        func.blocks.push(bb1);

        let pass = SimplifyCfg;
        assert!(pass.run(&mut func));
        assert_eq!(func.blocks.len(), 1);
        assert_eq!(func.blocks[0].id, BlockId(0));
    }

    #[test]
    fn test_merge_blocks() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb0 = BasicBlock::new(BlockId(0));
        bb0.instructions.push(MirInst::Const {
            dest: ValueId(0),
            value: ConstValue::I32(1),
        });
        bb0.terminator = Terminator::Branch { target: BlockId(1) };

        let mut bb1 = BasicBlock::new(BlockId(1));
        bb1.instructions.push(MirInst::Const {
            dest: ValueId(1),
            value: ConstValue::I32(2),
        });
        bb1.terminator = Terminator::Return;

        func.blocks.push(bb0);
        func.blocks.push(bb1);

        let pass = SimplifyCfg;
        assert!(pass.run(&mut func));
        assert_eq!(func.blocks.len(), 1);
        assert_eq!(func.blocks[0].instructions.len(), 2);
    }
}
