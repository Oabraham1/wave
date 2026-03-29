// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Sparse conditional constant propagation pass.
//!
//! Combines constant propagation with unreachable code detection. More
//! powerful than simple constant folding because it tracks which branches
//! are always taken and can prove code unreachable.

use std::collections::{HashMap, HashSet, VecDeque};

use super::pass::Pass;
use crate::hir::expr::BinOp;
use crate::mir::basic_block::Terminator;
use crate::mir::function::MirFunction;
use crate::mir::instruction::{ConstValue, MirInst};
use crate::mir::value::{BlockId, ValueId};

/// Lattice value for SCCP.
#[derive(Debug, Clone, PartialEq)]
enum Lattice {
    Top,
    Constant(i32),
    Bottom,
}

/// Sparse conditional constant propagation pass.
pub struct Sccp;

impl Pass for Sccp {
    fn name(&self) -> &str {
        "sccp"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let mut lattice: HashMap<ValueId, Lattice> = HashMap::new();
        let mut executable: HashSet<BlockId> = HashSet::new();
        let mut worklist: VecDeque<BlockId> = VecDeque::new();

        for param in &func.params {
            lattice.insert(param.value, Lattice::Top);
        }

        executable.insert(func.entry);
        worklist.push_back(func.entry);

        while let Some(bid) = worklist.pop_front() {
            let block = match func.block(bid) {
                Some(b) => b.clone(),
                None => continue,
            };

            for inst in &block.instructions {
                evaluate_instruction(inst, &mut lattice);
            }

            match &block.terminator {
                Terminator::Branch { target } => {
                    if executable.insert(*target) {
                        worklist.push_back(*target);
                    }
                }
                Terminator::CondBranch {
                    cond,
                    true_target,
                    false_target,
                } => {
                    match lattice.get(cond) {
                        Some(Lattice::Constant(v)) if *v != 0 => {
                            if executable.insert(*true_target) {
                                worklist.push_back(*true_target);
                            }
                        }
                        Some(Lattice::Constant(0)) => {
                            if executable.insert(*false_target) {
                                worklist.push_back(*false_target);
                            }
                        }
                        _ => {
                            if executable.insert(*true_target) {
                                worklist.push_back(*true_target);
                            }
                            if executable.insert(*false_target) {
                                worklist.push_back(*false_target);
                            }
                        }
                    }
                }
                Terminator::Return => {}
            }
        }

        let mut changed = false;

        for block in &mut func.blocks {
            for inst in &mut block.instructions {
                if let Some(dest) = inst.dest() {
                    if let Some(Lattice::Constant(v)) = lattice.get(&dest) {
                        if !matches!(inst, MirInst::Const { .. }) {
                            *inst = MirInst::Const {
                                dest,
                                value: ConstValue::I32(*v),
                            };
                            changed = true;
                        }
                    }
                }
            }
        }

        let original_count = func.blocks.len();
        func.blocks.retain(|b| executable.contains(&b.id) || b.id == func.entry);
        if func.blocks.len() != original_count {
            changed = true;
        }

        changed
    }
}

fn evaluate_instruction(inst: &MirInst, lattice: &mut HashMap<ValueId, Lattice>) {
    match inst {
        MirInst::Const { dest, value } => {
            let v = match value {
                ConstValue::I32(v) => *v,
                ConstValue::U32(v) => *v as i32,
                ConstValue::Bool(v) => i32::from(*v),
                ConstValue::F32(_) => return,
            };
            lattice.insert(*dest, Lattice::Constant(v));
        }
        MirInst::BinOp {
            dest, op, lhs, rhs, ..
        } => {
            let l = lattice.get(lhs).cloned().unwrap_or(Lattice::Bottom);
            let r = lattice.get(rhs).cloned().unwrap_or(Lattice::Bottom);

            let result = match (&l, &r) {
                (Lattice::Constant(a), Lattice::Constant(b)) => {
                    match op {
                        BinOp::Add => Lattice::Constant(a.wrapping_add(*b)),
                        BinOp::Sub => Lattice::Constant(a.wrapping_sub(*b)),
                        BinOp::Mul => Lattice::Constant(a.wrapping_mul(*b)),
                        BinOp::Lt => Lattice::Constant(i32::from(*a < *b)),
                        BinOp::Eq => Lattice::Constant(i32::from(*a == *b)),
                        _ => Lattice::Bottom,
                    }
                }
                _ => Lattice::Bottom,
            };
            lattice.insert(*dest, result);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::basic_block::{BasicBlock, Terminator};
    use crate::mir::types::MirType;

    #[test]
    fn test_sccp_folds_constants() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb = BasicBlock::new(BlockId(0));
        bb.instructions.push(MirInst::Const {
            dest: ValueId(0),
            value: ConstValue::I32(10),
        });
        bb.instructions.push(MirInst::Const {
            dest: ValueId(1),
            value: ConstValue::I32(20),
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

        let pass = Sccp;
        assert!(pass.run(&mut func));
        match &func.blocks[0].instructions[2] {
            MirInst::Const { value, .. } => assert_eq!(*value, ConstValue::I32(30)),
            other => panic!("expected Const, got {other:?}"),
        }
    }

    #[test]
    fn test_sccp_no_change_without_constants() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb = BasicBlock::new(BlockId(0));
        bb.instructions.push(MirInst::BinOp {
            dest: ValueId(2),
            op: BinOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: MirType::I32,
        });
        bb.terminator = Terminator::Return;
        func.blocks.push(bb);

        let pass = Sccp;
        assert!(!pass.run(&mut func));
    }
}
