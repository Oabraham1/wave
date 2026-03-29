// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Common subexpression elimination pass.
//!
//! For each instruction, checks if an equivalent computation with the
//! same operands already exists. If so, replaces the redundant computation
//! with a reference to the existing value.

use std::collections::HashMap;

use super::pass::Pass;
use crate::mir::function::MirFunction;
use crate::mir::instruction::MirInst;
use crate::mir::value::ValueId;

/// Common subexpression elimination pass.
pub struct Cse;

#[derive(Hash, PartialEq, Eq, Clone)]
struct ExprKey {
    op: u8,
    lhs: ValueId,
    rhs: ValueId,
}

impl Pass for Cse {
    fn name(&self) -> &str {
        "cse"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let mut available: HashMap<ExprKey, ValueId> = HashMap::new();
        let mut replacements: HashMap<ValueId, ValueId> = HashMap::new();
        let mut changed = false;

        for block in &mut func.blocks {
            for inst in &mut block.instructions {
                if let MirInst::BinOp {
                    dest, op, lhs, rhs, ..
                } = inst
                {
                    let actual_lhs = *replacements.get(lhs).unwrap_or(lhs);
                    let actual_rhs = *replacements.get(rhs).unwrap_or(rhs);
                    *lhs = actual_lhs;
                    *rhs = actual_rhs;

                    let key = ExprKey {
                        op: *op as u8,
                        lhs: actual_lhs,
                        rhs: actual_rhs,
                    };

                    if let Some(&existing) = available.get(&key) {
                        replacements.insert(*dest, existing);
                        *inst = MirInst::Const {
                            dest: *dest,
                            value: crate::mir::instruction::ConstValue::I32(0),
                        };
                        changed = true;
                    } else {
                        available.insert(key, *dest);
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
    use crate::mir::types::MirType;
    use crate::mir::value::BlockId;

    #[test]
    fn test_cse_eliminates_duplicate() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb = BasicBlock::new(BlockId(0));
        bb.instructions.push(MirInst::BinOp {
            dest: ValueId(2),
            op: BinOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: MirType::I32,
        });
        bb.instructions.push(MirInst::BinOp {
            dest: ValueId(3),
            op: BinOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: MirType::I32,
        });
        bb.terminator = Terminator::Return;
        func.blocks.push(bb);

        let pass = Cse;
        assert!(pass.run(&mut func));
    }

    #[test]
    fn test_cse_no_change_different_ops() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb = BasicBlock::new(BlockId(0));
        bb.instructions.push(MirInst::BinOp {
            dest: ValueId(2),
            op: BinOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: MirType::I32,
        });
        bb.instructions.push(MirInst::BinOp {
            dest: ValueId(3),
            op: BinOp::Sub,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: MirType::I32,
        });
        bb.terminator = Terminator::Return;
        func.blocks.push(bb);

        let pass = Cse;
        assert!(!pass.run(&mut func));
    }
}
