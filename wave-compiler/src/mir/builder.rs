// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! MIR construction helper for building functions incrementally.
//!
//! The builder provides a convenient API for creating basic blocks,
//! emitting instructions, and managing SSA value generation.

use super::basic_block::{BasicBlock, Terminator};
use super::function::{MirFunction, MirParam};
use super::instruction::{ConstValue, MirInst};
use super::types::MirType;
use super::value::{BlockId, IdGenerator, ValueId};
use crate::hir::expr::BinOp;
use crate::hir::types::AddressSpace;

/// Builder for constructing MIR functions incrementally.
pub struct MirBuilder {
    func: MirFunction,
    gen: IdGenerator,
    current_block: BlockId,
}

impl MirBuilder {
    /// Create a new builder for a function with the given name.
    #[must_use]
    pub fn new(name: String) -> Self {
        let mut gen = IdGenerator::new();
        let entry = gen.next_block();
        let func = MirFunction::new(name, entry);
        Self {
            func,
            gen,
            current_block: entry,
        }
    }

    /// Add a kernel parameter and return its SSA value.
    pub fn add_param(&mut self, name: String, ty: MirType) -> ValueId {
        let value = self.gen.next_value();
        self.func.params.push(MirParam {
            value,
            ty,
            name,
        });
        self.func.set_type(value, ty);
        value
    }

    /// Create a new basic block and return its ID.
    pub fn create_block(&mut self) -> BlockId {
        self.gen.next_block()
    }

    /// Switch to emitting instructions in the given block.
    pub fn switch_to_block(&mut self, block: BlockId) {
        self.current_block = block;
        if self.func.block(block).is_none() {
            self.func.blocks.push(BasicBlock::new(block));
        }
    }

    /// Generate a fresh SSA value ID.
    pub fn next_value(&mut self) -> ValueId {
        self.gen.next_value()
    }

    /// Emit a constant instruction.
    pub fn emit_const(&mut self, value: ConstValue) -> ValueId {
        let dest = self.gen.next_value();
        let ty = match &value {
            ConstValue::I32(_) | ConstValue::U32(_) => MirType::I32,
            ConstValue::F32(_) => MirType::F32,
            ConstValue::Bool(_) => MirType::Bool,
        };
        self.func.set_type(dest, ty);
        self.emit(MirInst::Const { dest, value });
        dest
    }

    /// Emit a binary operation.
    pub fn emit_binop(&mut self, op: BinOp, lhs: ValueId, rhs: ValueId, ty: MirType) -> ValueId {
        let result_ty = if op.is_comparison() {
            MirType::Bool
        } else {
            ty
        };
        let dest = self.gen.next_value();
        self.func.set_type(dest, result_ty);
        self.emit(MirInst::BinOp {
            dest,
            op,
            lhs,
            rhs,
            ty,
        });
        dest
    }

    /// Emit a load instruction.
    pub fn emit_load(
        &mut self,
        addr: ValueId,
        space: AddressSpace,
        ty: MirType,
    ) -> ValueId {
        let dest = self.gen.next_value();
        self.func.set_type(dest, ty);
        self.emit(MirInst::Load {
            dest,
            addr,
            space,
            ty,
        });
        dest
    }

    /// Emit a store instruction.
    pub fn emit_store(&mut self, addr: ValueId, value: ValueId, space: AddressSpace) {
        self.emit(MirInst::Store {
            addr,
            value,
            space,
        });
    }

    /// Emit an arbitrary instruction to the current block.
    pub fn emit(&mut self, inst: MirInst) {
        if let Some(block) = self.func.block_mut(self.current_block) {
            block.instructions.push(inst);
        }
    }

    /// Set the terminator for the current block.
    pub fn set_terminator(&mut self, term: Terminator) {
        if let Some(block) = self.func.block_mut(self.current_block) {
            block.terminator = term;
        }
    }

    /// Finalize and return the built function.
    pub fn finish(mut self) -> MirFunction {
        if self.func.block(self.func.entry).is_none() {
            self.func.blocks.insert(0, BasicBlock::new(self.func.entry));
        }
        self.func
    }

    /// Get a mutable reference to a block by ID.
    pub fn get_block_mut(&mut self, block: BlockId) -> Option<&mut BasicBlock> {
        self.func.block_mut(block)
    }

    /// Returns the current block ID.
    #[must_use]
    pub fn current_block(&self) -> BlockId {
        self.current_block
    }

    /// Returns the total number of values generated.
    #[must_use]
    pub fn value_count(&self) -> u32 {
        self.gen.value_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_simple_function() {
        let mut builder = MirBuilder::new("test".into());
        builder.switch_to_block(builder.current_block());
        let p0 = builder.add_param("a".into(), MirType::I32);
        let p1 = builder.add_param("b".into(), MirType::I32);
        let sum = builder.emit_binop(BinOp::Add, p0, p1, MirType::I32);
        builder.set_terminator(Terminator::Return);

        let func = builder.finish();
        assert_eq!(func.name, "test");
        assert_eq!(func.params.len(), 2);
        assert_eq!(func.block_count(), 1);
        assert_eq!(func.get_type(sum), Some(MirType::I32));
    }

    #[test]
    fn test_builder_multiple_blocks() {
        let mut builder = MirBuilder::new("test".into());
        let entry = builder.current_block();
        builder.switch_to_block(entry);

        let bb1 = builder.create_block();
        let bb2 = builder.create_block();

        let cond = builder.emit_const(ConstValue::Bool(true));
        builder.set_terminator(Terminator::CondBranch {
            cond,
            true_target: bb1,
            false_target: bb2,
        });

        builder.switch_to_block(bb1);
        builder.set_terminator(Terminator::Return);

        builder.switch_to_block(bb2);
        builder.set_terminator(Terminator::Return);

        let func = builder.finish();
        assert_eq!(func.block_count(), 3);
    }
}
