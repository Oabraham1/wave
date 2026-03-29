// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Strength reduction pass.
//!
//! Replaces expensive operations with cheaper equivalents:
//! multiply by power-of-2 → shift left, unsigned divide by power-of-2
//! → shift right, unsigned modulo by power-of-2 → bitwise AND.

use std::collections::HashMap;

use super::pass::Pass;
use crate::hir::expr::BinOp;
use crate::mir::function::MirFunction;
use crate::mir::instruction::{ConstValue, MirInst};
use crate::mir::value::ValueId;

/// Strength reduction pass.
pub struct StrengthReduce;

impl Pass for StrengthReduce {
    fn name(&self) -> &'static str {
        "strength_reduce"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let mut constants: HashMap<ValueId, u32> = HashMap::new();
        let mut changed = false;

        for block in &mut func.blocks {
            for inst in &block.instructions {
                if let MirInst::Const { dest, value } = inst {
                    match value {
                        ConstValue::I32(v) => {
                            constants.insert(*dest, u32::from_ne_bytes(v.to_ne_bytes()));
                        }
                        ConstValue::U32(v) => {
                            constants.insert(*dest, *v);
                        }
                        _ => {}
                    }
                }
            }

            let mut replacements: Vec<(usize, MirInst, MirInst)> = Vec::new();

            for (idx, inst) in block.instructions.iter().enumerate() {
                if let MirInst::BinOp {
                    dest,
                    op,
                    lhs,
                    rhs,
                    ty,
                } = inst
                {
                    if let Some(&rhs_val) = constants.get(rhs) {
                        if rhs_val.is_power_of_two() && rhs_val > 1 {
                            let shift = rhs_val.trailing_zeros();
                            let new_const_dest = ValueId(dest.0 + 10000);

                            match op {
                                BinOp::Mul => {
                                    replacements.push((
                                        idx,
                                        MirInst::BinOp {
                                            dest: *dest,
                                            op: BinOp::Shl,
                                            lhs: *lhs,
                                            rhs: new_const_dest,
                                            ty: *ty,
                                        },
                                        MirInst::Const {
                                            dest: new_const_dest,
                                            value: ConstValue::U32(shift),
                                        },
                                    ));
                                }
                                BinOp::Div | BinOp::FloorDiv => {
                                    replacements.push((
                                        idx,
                                        MirInst::BinOp {
                                            dest: *dest,
                                            op: BinOp::Shr,
                                            lhs: *lhs,
                                            rhs: new_const_dest,
                                            ty: *ty,
                                        },
                                        MirInst::Const {
                                            dest: new_const_dest,
                                            value: ConstValue::U32(shift),
                                        },
                                    ));
                                }
                                BinOp::Mod => {
                                    replacements.push((
                                        idx,
                                        MirInst::BinOp {
                                            dest: *dest,
                                            op: BinOp::BitAnd,
                                            lhs: *lhs,
                                            rhs: new_const_dest,
                                            ty: *ty,
                                        },
                                        MirInst::Const {
                                            dest: new_const_dest,
                                            value: ConstValue::U32(rhs_val - 1),
                                        },
                                    ));
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            for (idx, replacement, new_const) in replacements.into_iter().rev() {
                block.instructions[idx] = replacement;
                block.instructions.insert(idx, new_const);
                changed = true;
            }
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
