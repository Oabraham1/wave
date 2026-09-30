// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Common subexpression elimination pass. For each instruction, checks if an equivalent
//! computation with the same operands already exists. If so, replaces the redundant computation
//! with a reference to the existing value.

use std::collections::HashMap;

use super::pass::Pass;
use crate::mir::function::{resolve_value, MirFunction};
use crate::mir::instruction::MirInst;
use crate::mir::value::ValueId;

#[derive(Debug)]
/// Common subexpression elimination pass.
pub struct Cse;

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
struct ExprKey {
    op: u8,
    ty: crate::mir::types::MirType,
    lhs: ValueId,
    rhs: ValueId,
}

impl Pass for Cse {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let mut replacements = HashMap::new();
        for block in &mut func.blocks {
            let mut available = HashMap::new();
            block.instructions.retain_mut(|inst| {
                inst.map_operands(|value| resolve_value(value, &replacements));
                if let MirInst::BinOp {
                    dest,
                    op,
                    lhs,
                    rhs,
                    ty,
                } = inst
                {
                    let key = ExprKey {
                        op: *op as u8,
                        ty: *ty,
                        lhs: *lhs,
                        rhs: *rhs,
                    };
                    if let Some(&existing) = available.get(&key) {
                        replacements.insert(*dest, existing);
                        return false;
                    }
                    available.insert(key, *dest);
                }
                true
            });
        }
        func.replace_uses(&replacements);
        !replacements.is_empty()
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
