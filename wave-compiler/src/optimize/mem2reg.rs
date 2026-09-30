// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Forward private stores to matching loads within a block. Memory writes and calls invalidate
//! available values.

use std::collections::HashMap;

use super::pass::Pass;
use crate::hir::types::AddressSpace;
use crate::mir::function::{resolve_value, MirFunction};
use crate::mir::instruction::MirInst;
use crate::mir::value::ValueId;

#[derive(Debug)]
/// Memory-to-register promotion pass.
pub struct Mem2Reg;

impl Pass for Mem2Reg {
    fn name(&self) -> &'static str {
        "mem2reg"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let mut replacements = HashMap::new();
        for block in &mut func.blocks {
            let mut local_stores: HashMap<ValueId, ValueId> = HashMap::new();
            block.instructions.retain_mut(|inst| {
                inst.map_operands(|value| resolve_value(value, &replacements));
                match inst {
                    MirInst::Store { addr, value, space } => {
                        local_stores.clear();
                        if *space == AddressSpace::Private {
                            local_stores.insert(*addr, *value);
                        }
                    }
                    MirInst::Load {
                        dest,
                        addr,
                        space: AddressSpace::Private,
                        ty,
                    } => {
                        if let Some(&stored) = local_stores.get(addr) {
                            if func.value_types.get(&stored) == Some(ty) {
                                replacements.insert(*dest, stored);
                                return false;
                            }
                        }
                    }
                    other if other.has_side_effects() => local_stores.clear(),
                    _ => {}
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

        func.set_type(ValueId(1), MirType::I32);
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

        func.set_type(ValueId(1), MirType::I32);
        let pass = Mem2Reg;
        assert!(!pass.run(&mut func));
    }
}
