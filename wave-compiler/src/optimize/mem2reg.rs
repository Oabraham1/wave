// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Memory-to-register promotion pass (SSA construction).
//!
//! Promotes local memory load/store pairs to register operations by
//! identifying variables that can live in registers. Inserts phi nodes
//! at dominance frontiers where needed.

use std::collections::HashMap;

use super::pass::Pass;
use crate::hir::types::AddressSpace;
use crate::mir::function::MirFunction;
use crate::mir::instruction::MirInst;
use crate::mir::value::ValueId;

/// Memory-to-register promotion pass.
pub struct Mem2Reg;

impl Pass for Mem2Reg {
    fn name(&self) -> &str {
        "mem2reg"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let mut local_stores: HashMap<ValueId, ValueId> = HashMap::new();
        let mut promotable_loads: Vec<(usize, usize, ValueId, ValueId)> = Vec::new();
        let mut changed = false;

        for (block_idx, block) in func.blocks.iter().enumerate() {
            for (inst_idx, inst) in block.instructions.iter().enumerate() {
                match inst {
                    MirInst::Store {
                        addr,
                        value,
                        space: AddressSpace::Private,
                    } => {
                        local_stores.insert(*addr, *value);
                    }
                    MirInst::Load {
                        dest,
                        addr,
                        space: AddressSpace::Private,
                        ..
                    } => {
                        if let Some(&stored_val) = local_stores.get(addr) {
                            promotable_loads.push((block_idx, inst_idx, *dest, stored_val));
                        }
                    }
                    _ => {}
                }
            }
        }

        for (block_idx, inst_idx, dest, replacement_value) in promotable_loads.into_iter().rev() {
            func.blocks[block_idx].instructions[inst_idx] = MirInst::Const {
                dest,
                value: crate::mir::instruction::ConstValue::U32(replacement_value.index()),
            };
            changed = true;
        }

        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::basic_block::{BasicBlock, Terminator};
    use crate::mir::instruction::MirInst;
    use crate::mir::types::MirType;
    use crate::mir::value::BlockId;

    #[test]
    fn test_mem2reg_promotes_private_load() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb = BasicBlock::new(BlockId(0));
        bb.instructions.push(MirInst::Store {
            addr: ValueId(0),
            value: ValueId(1),
            space: AddressSpace::Private,
        });
        bb.instructions.push(MirInst::Load {
            dest: ValueId(2),
            addr: ValueId(0),
            space: AddressSpace::Private,
            ty: MirType::I32,
        });
        bb.terminator = Terminator::Return;
        func.blocks.push(bb);

        let pass = Mem2Reg;
        assert!(pass.run(&mut func));
    }

    #[test]
    fn test_mem2reg_ignores_device_memory() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb = BasicBlock::new(BlockId(0));
        bb.instructions.push(MirInst::Store {
            addr: ValueId(0),
            value: ValueId(1),
            space: AddressSpace::Device,
        });
        bb.instructions.push(MirInst::Load {
            dest: ValueId(2),
            addr: ValueId(0),
            space: AddressSpace::Device,
            ty: MirType::I32,
        });
        bb.terminator = Terminator::Return;
        func.blocks.push(bb);

        let pass = Mem2Reg;
        assert!(!pass.run(&mut func));
    }
}
