// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Constant folding and propagation pass.
//!
//! Evaluates operations on constants at compile time. When both operands
//! of a binary operation are constants, the result is computed and the
//! instruction is replaced with a constant.

use std::collections::HashMap;

use super::pass::Pass;
use crate::hir::expr::BinOp;
use crate::mir::function::MirFunction;
use crate::mir::instruction::{ConstValue, MirInst};
use crate::mir::value::ValueId;

/// Constant folding pass.
pub struct ConstantFold;

impl Pass for ConstantFold {
    fn name(&self) -> &str {
        "constant_fold"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let mut constants: HashMap<ValueId, i32> = HashMap::new();
        let mut changed = false;

        for block in &mut func.blocks {
            for inst in &mut block.instructions {
                match inst {
                    MirInst::Const { dest, value } => {
                        if let ConstValue::I32(v) = value {
                            constants.insert(*dest, *v);
                        }
                        if let ConstValue::U32(v) = value {
                            constants.insert(*dest, *v as i32);
                        }
                    }
                    MirInst::BinOp {
                        dest, op, lhs, rhs, ..
                    } => {
                        if let (Some(&l), Some(&r)) = (constants.get(lhs), constants.get(rhs)) {
                            if let Some(result) = fold_binop(*op, l, r) {
                                constants.insert(*dest, result);
                                *inst = MirInst::Const {
                                    dest: *dest,
                                    value: ConstValue::I32(result),
                                };
                                changed = true;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        changed
    }
}

fn fold_binop(op: BinOp, lhs: i32, rhs: i32) -> Option<i32> {
    match op {
        BinOp::Add => Some(lhs.wrapping_add(rhs)),
        BinOp::Sub => Some(lhs.wrapping_sub(rhs)),
        BinOp::Mul => Some(lhs.wrapping_mul(rhs)),
        BinOp::Div | BinOp::FloorDiv => {
            if rhs == 0 {
                None
            } else {
                Some(lhs.wrapping_div(rhs))
            }
        }
        BinOp::Mod => {
            if rhs == 0 {
                None
            } else {
                Some(lhs.wrapping_rem(rhs))
            }
        }
        BinOp::BitAnd => Some(lhs & rhs),
        BinOp::BitOr => Some(lhs | rhs),
        BinOp::BitXor => Some(lhs ^ rhs),
        BinOp::Shl => Some(lhs.wrapping_shl(rhs as u32)),
        BinOp::Shr => Some((lhs as u32).wrapping_shr(rhs as u32) as i32),
        BinOp::Eq => Some(i32::from(lhs == rhs)),
        BinOp::Ne => Some(i32::from(lhs != rhs)),
        BinOp::Lt => Some(i32::from(lhs < rhs)),
        BinOp::Le => Some(i32::from(lhs <= rhs)),
        BinOp::Gt => Some(i32::from(lhs > rhs)),
        BinOp::Ge => Some(i32::from(lhs >= rhs)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::basic_block::{BasicBlock, Terminator};
    use crate::mir::types::MirType;
    use crate::mir::value::BlockId;

    #[test]
    fn test_fold_addition() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb = BasicBlock::new(BlockId(0));
        bb.instructions.push(MirInst::Const {
            dest: ValueId(0),
            value: ConstValue::I32(5),
        });
        bb.instructions.push(MirInst::Const {
            dest: ValueId(1),
            value: ConstValue::I32(3),
        });
        bb.instructions.push(MirInst::BinOp {
            dest: ValueId(2),
            op: BinOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: MirType::I32,
        });
        bb.terminator = Terminator::Return;
        func.blocks.push(bb);

        let pass = ConstantFold;
        assert!(pass.run(&mut func));

        match &func.blocks[0].instructions[2] {
            MirInst::Const { value, .. } => {
                assert_eq!(*value, ConstValue::I32(8));
            }
            other => panic!("expected Const, got {other:?}"),
        }
    }

    #[test]
    fn test_no_fold_non_constant() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb = BasicBlock::new(BlockId(0));
        bb.instructions.push(MirInst::Const {
            dest: ValueId(0),
            value: ConstValue::I32(5),
        });
        bb.instructions.push(MirInst::BinOp {
            dest: ValueId(2),
            op: BinOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: MirType::I32,
        });
        bb.terminator = Terminator::Return;
        func.blocks.push(bb);

        let pass = ConstantFold;
        assert!(!pass.run(&mut func));
    }
}
