// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Reduce integer multiplication by powers of two to shifts.

use std::collections::HashMap;

use super::pass::Pass;
use crate::hir::expr::BinOp;
use crate::mir::function::MirFunction;
use crate::mir::instruction::{ConstValue, MirInst};
use crate::mir::value::ValueId;

#[derive(Debug)]
/// Strength reduction pass.
pub struct StrengthReduce;

impl Pass for StrengthReduce {
    fn name(&self) -> &'static str {
        "strength_reduce"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let Some(mut next_value) = func.next_value_id() else {
            return false;
        };
        let constants: HashMap<ValueId, u32> = func
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .filter_map(|inst| match inst {
                MirInst::Const {
                    dest,
                    value: ConstValue::I32(value),
                } => Some((*dest, value.cast_unsigned())),
                MirInst::Const {
                    dest,
                    value: ConstValue::U32(value),
                } => Some((*dest, *value)),
                _ => None,
            })
            .collect();
        let mut changed = false;
        for block in &mut func.blocks {
            let mut instructions = Vec::with_capacity(block.instructions.len());
            for mut inst in std::mem::take(&mut block.instructions) {
                if let MirInst::BinOp {
                    op: BinOp::Mul,
                    rhs,
                    ty,
                    ..
                } = &inst
                {
                    if let Some(&value) = constants.get(rhs).filter(|_| ty.is_integer()) {
                        if value.is_power_of_two() && value > 1 {
                            if let Some(next) = next_value.checked_add(1) {
                                let shift = ValueId(next_value);
                                next_value = next;
                                instructions.push(MirInst::Const {
                                    dest: shift,
                                    value: ConstValue::U32(value.trailing_zeros()),
                                });
                                func.value_types
                                    .insert(shift, crate::mir::types::MirType::I32);
                                if let MirInst::BinOp { op, rhs, .. } = &mut inst {
                                    *op = BinOp::Shl;
                                    *rhs = shift;
                                }
                                changed = true;
                            }
                        }
                    }
                }
                instructions.push(inst);
            }
            block.instructions = instructions;
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::basic_block::{BasicBlock, Terminator};
    use crate::mir::types::MirType;
    use crate::mir::value::BlockId;

    #[test]
    fn test_strength_reduce_mul_by_power_of_two() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb = BasicBlock::new(BlockId(0));
        bb.instructions.push(MirInst::Const {
            dest: ValueId(1),
            value: ConstValue::U32(8),
        });
        bb.instructions.push(MirInst::BinOp {
            dest: ValueId(2),
            op: BinOp::Mul,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: MirType::I32,
        });
        bb.terminator = Terminator::Return;
        func.blocks.push(bb);

        let pass = StrengthReduce;
        assert!(pass.run(&mut func));
        let has_shl = func.blocks[0]
            .instructions
            .iter()
            .any(|i| matches!(i, MirInst::BinOp { op: BinOp::Shl, .. }));
        assert!(has_shl);
    }

    #[test]
    fn test_no_reduce_non_power_of_two() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb = BasicBlock::new(BlockId(0));
        bb.instructions.push(MirInst::Const {
            dest: ValueId(1),
            value: ConstValue::U32(7),
        });
        bb.instructions.push(MirInst::BinOp {
            dest: ValueId(2),
            op: BinOp::Mul,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: MirType::I32,
        });
        bb.terminator = Terminator::Return;
        func.blocks.push(bb);

        let pass = StrengthReduce;
        assert!(!pass.run(&mut func));
    }
}
