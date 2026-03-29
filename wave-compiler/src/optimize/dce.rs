// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Dead code elimination pass.
//!
//! Marks all instructions whose results are used. Removes unmarked
//! instructions that have no side effects (stores, barriers, atomics
//! are always considered live).

use std::collections::HashSet;

use super::pass::Pass;
use crate::mir::function::MirFunction;
use crate::mir::value::ValueId;

/// Dead code elimination pass.
pub struct Dce;

impl Pass for Dce {
    fn name(&self) -> &'static str {
        "dce"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let used_values = collect_used_values(func);
        let mut changed = false;

        for block in &mut func.blocks {
            let original_len = block.instructions.len();
            block.instructions.retain(|inst| {
                if inst.has_side_effects() {
                    return true;
                }
                match inst.dest() {
                    Some(dest) => used_values.contains(&dest),
                    None => true,
                }
            });
            if block.instructions.len() != original_len {
                changed = true;
            }
        }

        changed
    }
}

fn collect_used_values(func: &MirFunction) -> HashSet<ValueId> {
    let mut used = HashSet::new();
    for block in &func.blocks {
        for phi in &block.phis {
            for (_, val) in &phi.incoming {
                used.insert(*val);
            }
        }
        for inst in &block.instructions {
            for operand in inst.operands() {
                used.insert(operand);
            }
        }
        for operand in block.terminator.operands() {
            used.insert(operand);
        }
    }
    used
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
        assert!(func.blocks[0].instructions.is_empty());
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
