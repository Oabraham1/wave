// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! CFG simplification pass. Merges basic blocks connected by unconditional branches when the
//! target has a single predecessor. Removes empty blocks and unreachable blocks to simplify the
//! control flow graph.

use std::collections::{HashMap, HashSet};

use super::pass::Pass;
use crate::mir::basic_block::Terminator;
use crate::mir::function::MirFunction;

#[derive(Debug)]
/// CFG simplification pass.
pub struct SimplifyCfg;

impl Pass for SimplifyCfg {
    fn name(&self) -> &'static str {
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
    for block in &mut func.blocks {
        for phi in &mut block.phis {
            phi.incoming.retain(|(bid, _)| reachable.contains(bid));
        }
    }
    func.blocks.len() != original_count
}

fn merge_single_predecessor_blocks(func: &mut MirFunction) -> bool {
    let mut changed = false;
    loop {
        let preds = func.predecessors();
        let candidate = func.blocks.iter().enumerate().find_map(|(source, block)| {
            let Terminator::Branch { target } = block.terminator else {
                return None;
            };
            if target == block.id
                || target == func.entry
                || preds.get(&target)?.as_slice() != [block.id]
            {
                return None;
            }
            let target_index = func.blocks.iter().position(|other| other.id == target)?;
            let target_block = &func.blocks[target_index];
            if target_block
                .phis
                .iter()
                .any(|phi| phi.incoming.len() != 1 || phi.incoming[0].0 != block.id)
            {
                return None;
            }
            Some((source, target_index))
        });
        let Some((source, target)) = candidate else {
            break;
        };
        let target_block = func.blocks.remove(target);
        let source = if target < source { source - 1 } else { source };
        let source_id = func.blocks[source].id;
        let replacements: HashMap<_, _> = target_block
            .phis
            .iter()
            .map(|phi| (phi.dest, phi.incoming[0].1))
            .collect();
        func.blocks[source]
            .instructions
            .extend(target_block.instructions);
        func.blocks[source].terminator = target_block.terminator;
        for block in &mut func.blocks {
            for phi in &mut block.phis {
                for (predecessor, _) in &mut phi.incoming {
                    if *predecessor == target_block.id {
                        *predecessor = source_id;
                    }
                }
            }
        }
        func.replace_uses(&replacements);
        changed = true;
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
