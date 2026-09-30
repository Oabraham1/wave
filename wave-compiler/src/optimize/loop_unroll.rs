// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Fully unroll single-block loops with at most four statically determined iterations.

use super::pass::Pass;
use crate::mir::basic_block::{BasicBlock, Terminator};
use crate::mir::function::{resolve_value, MirFunction};
use crate::mir::instruction::{ConstValue, MirInst};
use crate::mir::value::{BlockId, ValueId};
use std::collections::HashMap;

const MAX_UNROLL_FACTOR: usize = 4;
const MAX_UNROLLED_BODY_SIZE: usize = 128;

/// Loop unrolling pass.
#[derive(Debug)]
pub struct LoopUnroll;

#[derive(Debug)]
struct Unrolled {
    instructions: Vec<MirInst>,
    replacements: HashMap<ValueId, ValueId>,
    definitions: Vec<(ValueId, ValueId)>,
    exit: BlockId,
    next_value: u32,
}

impl Pass for LoopUnroll {
    fn name(&self) -> &'static str {
        "loop_unroll"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let Some(mut next_value) = func.next_value_id() else {
            return false;
        };
        let constants: HashMap<ValueId, i32> = func
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .filter_map(|inst| constant_value(inst, &HashMap::new()))
            .collect();
        let predecessors = func.predecessors();
        let mut changed = false;
        for index in 0..func.blocks.len() {
            let block = &func.blocks[index];
            let outside: Vec<_> = predecessors
                .get(&block.id)
                .into_iter()
                .flatten()
                .copied()
                .filter(|&bid| bid != block.id)
                .collect();
            let [preheader] = outside.as_slice() else {
                continue;
            };
            let Some(unrolled) = unroll(block, *preheader, &constants, next_value) else {
                continue;
            };
            next_value = unrolled.next_value;
            for (source, dest) in unrolled.definitions {
                if let Some(&ty) = func.value_types.get(&source) {
                    func.value_types.insert(dest, ty);
                }
            }
            func.blocks[index].instructions = unrolled.instructions;
            func.blocks[index].phis.clear();
            func.blocks[index].terminator = Terminator::Branch {
                target: unrolled.exit,
            };
            func.replace_uses(&unrolled.replacements);
            changed = true;
        }
        changed
    }
}

fn unroll(
    block: &BasicBlock,
    preheader: BlockId,
    constants: &HashMap<ValueId, i32>,
    mut next_value: u32,
) -> Option<Unrolled> {
    let Terminator::CondBranch {
        cond,
        true_target,
        false_target,
    } = block.terminator
    else {
        return None;
    };
    let (exit, repeat_on_true) = match (true_target == block.id, false_target == block.id) {
        (true, false) => (false_target, true),
        (false, true) => (true_target, false),
        _ => return None,
    };
    if block.instructions.len() > MAX_UNROLLED_BODY_SIZE / MAX_UNROLL_FACTOR {
        return None;
    }
    let mut replacements = HashMap::new();
    for phi in &block.phis {
        if phi.incoming.len() != 2 || !phi.incoming.iter().any(|&(bid, _)| bid == block.id) {
            return None;
        }
        let &(_, initial) = phi.incoming.iter().find(|&&(bid, _)| bid == preheader)?;
        replacements.insert(phi.dest, initial);
    }
    let mut values = constants.clone();
    let mut instructions = Vec::new();
    let mut definitions = Vec::new();
    for _ in 0..MAX_UNROLL_FACTOR {
        for original in &block.instructions {
            let mut inst = original.clone();
            inst.map_operands(|value| resolve_value(value, &replacements));
            if let Some(source) = original.dest() {
                let dest = ValueId(next_value);
                next_value = next_value.checked_add(1)?;
                inst.map_dest(|_| dest);
                replacements.insert(source, dest);
                definitions.push((source, dest));
            }
            if let Some((dest, value)) = constant_value(&inst, &values) {
                values.insert(dest, value);
            }
            instructions.push(inst);
        }
        let condition = *values.get(&resolve_value(cond, &replacements))? != 0;
        if condition != repeat_on_true {
            return Some(Unrolled {
                instructions,
                replacements,
                definitions,
                exit,
                next_value,
            });
        }
        let carried: Vec<_> = block
            .phis
            .iter()
            .map(|phi| {
                let &(_, value) = phi.incoming.iter().find(|&&(bid, _)| bid == block.id)?;
                Some((phi.dest, resolve_value(value, &replacements)))
            })
            .collect::<Option<_>>()?;
        replacements.extend(carried);
    }
    None
}

fn constant_value(inst: &MirInst, values: &HashMap<ValueId, i32>) -> Option<(ValueId, i32)> {
    match inst {
        MirInst::Const { dest, value } => Some((
            *dest,
            match value {
                ConstValue::I32(value) => *value,
                ConstValue::U32(value) => value.cast_signed(),
                ConstValue::Bool(value) => i32::from(*value),
                ConstValue::F32(_) => return None,
            },
        )),
        MirInst::BinOp {
            dest,
            op,
            lhs,
            rhs,
            ty,
        } if !ty.is_float() => Some((
            *dest,
            super::constant_fold::fold_binop(*op, *values.get(lhs)?, *values.get(rhs)?)?,
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::basic_block::{BasicBlock, Terminator};
    use crate::mir::instruction::{ConstValue, MirInst};
    use crate::mir::value::{BlockId, ValueId};

    #[test]
    fn test_loop_unroll_no_loops() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let bb = BasicBlock::new(BlockId(0));
        func.blocks.push(bb);

        let pass = LoopUnroll;
        assert!(!pass.run(&mut func));
    }

    #[test]
    fn test_loop_unroll_simple_loop() {
        let mut func = MirFunction::new("test".into(), BlockId(0));

        let mut bb0 = BasicBlock::new(BlockId(0));
        bb0.terminator = Terminator::Branch { target: BlockId(1) };

        let mut bb1 = BasicBlock::new(BlockId(1));
        bb1.instructions.push(MirInst::Const {
            dest: ValueId(0),
            value: ConstValue::I32(1),
        });
        bb1.terminator = Terminator::CondBranch {
            cond: ValueId(0),
            true_target: BlockId(2),
            false_target: BlockId(3),
        };

        let mut bb2 = BasicBlock::new(BlockId(2));
        bb2.terminator = Terminator::Branch { target: BlockId(1) };

        let bb3 = BasicBlock::new(BlockId(3));

        func.blocks.push(bb0);
        func.blocks.push(bb1);
        func.blocks.push(bb2);
        func.blocks.push(bb3);

        let pass = LoopUnroll;
        pass.run(&mut func);
        assert!(func.blocks.len() >= 4);
    }
}
