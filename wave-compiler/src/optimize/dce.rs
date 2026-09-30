// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Remove unused SSA instructions and phi nodes. Terminators and side effects keep their
//! transitive dependencies live.

use std::collections::{HashMap, HashSet};

use super::pass::Pass;
use crate::mir::function::MirFunction;
use crate::mir::value::ValueId;

/// Dead code elimination pass.
#[derive(Debug)]
pub struct Dce;

impl Pass for Dce {
    fn name(&self) -> &'static str {
        "dce"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let live = live_values(func);
        let mut changed = false;
        for block in &mut func.blocks {
            let original_len = block.instructions.len() + block.phis.len();
            block.instructions.retain(|inst| {
                inst.has_side_effects() || inst.dest().is_none_or(|dest| live.contains(&dest))
            });
            block.phis.retain(|phi| live.contains(&phi.dest));
            changed |= original_len != block.instructions.len() + block.phis.len();
        }
        changed
    }
}

fn live_values(func: &MirFunction) -> HashSet<ValueId> {
    let mut instructions = HashMap::new();
    let mut phis = HashMap::new();
    let mut pending = Vec::new();
    for block in &func.blocks {
        for phi in &block.phis {
            phis.insert(phi.dest, phi);
        }
        for inst in &block.instructions {
            if let Some(dest) = inst.dest() {
                instructions.insert(dest, inst);
            }
            if inst.has_side_effects() || inst.dest().is_none() {
                pending.extend(inst.operand_iter());
            }
        }
        pending.extend(block.terminator.operand_iter());
    }

    let mut live = HashSet::new();
    while let Some(value) = pending.pop() {
        if !live.insert(value) {
            continue;
        }
        if let Some(inst) = instructions.get(&value) {
            pending.extend(inst.operand_iter());
        } else if let Some(phi) = phis.get(&value) {
            pending.extend(phi.incoming.iter().map(|&(_, value)| value));
        }
    }
    live
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
    fn test_dce_removes_dead_code() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb = BasicBlock::new(BlockId(0));
        bb.instructions.push(MirInst::Const {
            dest: ValueId(0),
            value: ConstValue::I32(42),
        });
        bb.instructions.push(MirInst::Const {
            dest: ValueId(1),
            value: ConstValue::I32(99),
        });
        bb.terminator = Terminator::Return;
        func.blocks.push(bb);

        let pass = Dce;
        let changed = pass.run(&mut func);
        assert!(changed);
        assert_eq!(func.blocks[0].instructions, [] as [MirInst; 0]);
    }

    #[test]
    fn test_dce_preserves_used_values() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb = BasicBlock::new(BlockId(0));
        bb.instructions.push(MirInst::Const {
            dest: ValueId(0),
            value: ConstValue::I32(1),
        });
        bb.instructions.push(MirInst::Const {
            dest: ValueId(1),
            value: ConstValue::I32(2),
        });
        bb.instructions.push(MirInst::BinOp {
            dest: ValueId(2),
            op: BinOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: MirType::I32,
        });
        bb.terminator = Terminator::CondBranch {
            cond: ValueId(2),
            true_target: BlockId(0),
            false_target: BlockId(0),
        };
        func.blocks.push(bb);

        let pass = Dce;
        let changed = pass.run(&mut func);
        assert!(!changed);
        assert_eq!(func.blocks[0].instructions.len(), 3);
    }
}
