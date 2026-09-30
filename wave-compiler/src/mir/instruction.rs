// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Typed MIR instructions in SSA form, covering arithmetic, memory, comparisons, conversions,
//! and GPU operations.

use super::types::MirType;
use super::value::ValueId;
use crate::hir::expr::{BinOp, BuiltinFunc, MemoryScope, ShuffleMode};
use crate::hir::types::AddressSpace;

/// Atomic memory operation kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomicOp {
    /// Atomic addition.
    Add,
    /// Atomic minimum.
    Min,
    /// Atomic maximum.
    Max,
}

/// Constant value for MIR const instructions.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstValue {
    /// Integer constant.
    I32(i32),
    /// Unsigned integer constant.
    U32(u32),
    /// Float constant.
    F32(f32),
    /// Boolean constant.
    Bool(bool),
}

impl ConstValue {
    /// Returns the bits of this constant as a u32.
    #[must_use]
    pub fn to_bits(&self) -> u32 {
        match self {
            Self::I32(v) => (*v).cast_unsigned(),
            Self::U32(v) => *v,
            Self::F32(v) => v.to_bits(),
            Self::Bool(v) => u32::from(*v),
        }
    }
}

/// A MIR instruction in SSA form.
#[derive(Debug, Clone, PartialEq)]
pub enum MirInst {
    /// Binary arithmetic/logic operation.
    BinOp {
        /// Destination value.
        dest: ValueId,
        /// Operator.
        op: BinOp,
        /// Left operand.
        lhs: ValueId,
        /// Right operand.
        rhs: ValueId,
        /// Result type.
        ty: MirType,
    },
    /// Unary operation.
    UnaryOp {
        /// Destination value.
        dest: ValueId,
        /// Operator.
        op: crate::hir::expr::UnaryOp,
        /// Operand.
        operand: ValueId,
        /// Result type.
        ty: MirType,
    },
    /// Load from memory.
    Load {
        /// Destination value.
        dest: ValueId,
        /// Address to load from.
        addr: ValueId,
        /// Address space.
        space: AddressSpace,
        /// Loaded value type.
        ty: MirType,
    },
    /// Store to memory.
    Store {
        /// Address to store to.
        addr: ValueId,
        /// Value to store.
        value: ValueId,
        /// Address space.
        space: AddressSpace,
    },
    /// Built-in function call.
    Call {
        /// Optional destination value.
        dest: Option<ValueId>,
        /// Function being called.
        func: BuiltinFunc,
        /// Arguments.
        args: Vec<ValueId>,
    },
    /// Type cast/conversion.
    Cast {
        /// Destination value.
        dest: ValueId,
        /// Source value.
        value: ValueId,
        /// Source type.
        from: MirType,
        /// Target type.
        to: MirType,
    },
    /// Constant value.
    Const {
        /// Destination value.
        dest: ValueId,
        /// The constant.
        value: ConstValue,
    },
    /// Wave shuffle operation.
    Shuffle {
        /// Destination value.
        dest: ValueId,
        /// Value to shuffle.
        value: ValueId,
        /// Target lane/offset.
        lane: ValueId,
        /// Shuffle mode.
        mode: ShuffleMode,
    },
    /// Read a special register, `thread_id`, `workgroup_id`, etc..
    ReadSpecialReg {
        /// Destination value.
        dest: ValueId,
        /// Special register index.
        sr_index: u8,
    },
    /// Atomic read-modify-write.
    AtomicRmw {
        /// Destination value, old value.
        dest: ValueId,
        /// Address.
        addr: ValueId,
        /// Operand value.
        value: ValueId,
        /// Atomic operation.
        op: AtomicOp,
        /// Memory scope.
        scope: MemoryScope,
    },
    /// Workgroup barrier.
    Barrier,
    /// Memory fence.
    Fence {
        /// Scope of the fence.
        scope: MemoryScope,
    },
}

impl MirInst {
    /// Returns the destination value ID if this instruction produces one.
    #[must_use]
    pub fn dest(&self) -> Option<ValueId> {
        match self {
            Self::BinOp { dest, .. }
            | Self::UnaryOp { dest, .. }
            | Self::Load { dest, .. }
            | Self::Cast { dest, .. }
            | Self::Const { dest, .. }
            | Self::Shuffle { dest, .. }
            | Self::ReadSpecialReg { dest, .. }
            | Self::AtomicRmw { dest, .. } => Some(*dest),
            Self::Call { dest, .. } => *dest,
            Self::Store { .. } | Self::Barrier | Self::Fence { .. } => None,
        }
    }

    /// Returns all value IDs used as operands by this instruction.
    #[must_use]
    pub fn operands(&self) -> Vec<ValueId> {
        self.operand_iter().collect()
    }

    /// Iterate over source operands without allocating.
    pub(crate) fn operand_iter(&self) -> impl Iterator<Item = ValueId> + '_ {
        let operands = match self {
            Self::BinOp { lhs, rhs, .. } => [Some(*lhs), Some(*rhs)],
            Self::UnaryOp { operand, .. } => [Some(*operand), None],
            Self::Load { addr, .. } => [Some(*addr), None],
            Self::Store { addr, value, .. } | Self::AtomicRmw { addr, value, .. } => {
                [Some(*addr), Some(*value)]
            }
            Self::Cast { value, .. } => [Some(*value), None],
            Self::Call { .. }
            | Self::Const { .. }
            | Self::ReadSpecialReg { .. }
            | Self::Barrier
            | Self::Fence { .. } => [None, None],
            Self::Shuffle { value, lane, .. } => [Some(*value), Some(*lane)],
        };
        let args = match self {
            Self::Call { args, .. } => args.as_slice(),
            _ => &[],
        };
        operands.into_iter().flatten().chain(args.iter().copied())
    }

    /// Rewrite operand references, leaving the destination unchanged.
    pub(crate) fn map_operands(&mut self, mut map: impl FnMut(ValueId) -> ValueId) {
        match self {
            Self::BinOp { lhs, rhs, .. } => {
                *lhs = map(*lhs);
                *rhs = map(*rhs);
            }
            Self::UnaryOp { operand, .. } => *operand = map(*operand),
            Self::Load { addr, .. } => *addr = map(*addr),
            Self::Store { addr, value, .. } | Self::AtomicRmw { addr, value, .. } => {
                *addr = map(*addr);
                *value = map(*value);
            }
            Self::Call { args, .. } => args.iter_mut().for_each(|value| *value = map(*value)),
            Self::Cast { value, .. } => *value = map(*value),
            Self::Shuffle { value, lane, .. } => {
                *value = map(*value);
                *lane = map(*lane);
            }
            Self::Const { .. }
            | Self::ReadSpecialReg { .. }
            | Self::Barrier
            | Self::Fence { .. } => {}
        }
    }

    /// Rewrite the destination, leaving operands unchanged.
    pub(crate) fn map_dest(&mut self, mut map: impl FnMut(ValueId) -> ValueId) {
        match self {
            Self::BinOp { dest, .. }
            | Self::UnaryOp { dest, .. }
            | Self::Load { dest, .. }
            | Self::Cast { dest, .. }
            | Self::Const { dest, .. }
            | Self::Shuffle { dest, .. }
            | Self::ReadSpecialReg { dest, .. }
            | Self::AtomicRmw { dest, .. } => *dest = map(*dest),
            Self::Call { dest, .. } => *dest = dest.map(map),
            Self::Store { .. } | Self::Barrier | Self::Fence { .. } => {}
        }
    }

    /// Returns true if this instruction has side effects.
    #[must_use]
    pub fn has_side_effects(&self) -> bool {
        matches!(
            self,
            Self::Store { .. }
                | Self::AtomicRmw { .. }
                | Self::Barrier
                | Self::Fence { .. }
                | Self::Call { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_const_value_bits() {
        assert_eq!(ConstValue::I32(42).to_bits(), 42);
        assert_eq!(ConstValue::U32(0xFF).to_bits(), 0xFF);
        assert_eq!(ConstValue::F32(1.0).to_bits(), 0x3F80_0000);
        assert_eq!(ConstValue::Bool(true).to_bits(), 1);
        assert_eq!(ConstValue::Bool(false).to_bits(), 0);
    }

    #[test]
    fn test_instruction_dest_and_operands() {
        let inst = MirInst::BinOp {
            dest: ValueId(3),
            op: BinOp::Add,
            lhs: ValueId(1),
            rhs: ValueId(2),
            ty: MirType::I32,
        };
        assert_eq!(inst.dest(), Some(ValueId(3)));
        assert_eq!(inst.operands(), vec![ValueId(1), ValueId(2)]);
        assert!(!inst.has_side_effects());
    }

    #[test]
    fn test_store_has_side_effects() {
        let inst = MirInst::Store {
            addr: ValueId(0),
            value: ValueId(1),
            space: AddressSpace::Device,
        };
        assert!(inst.has_side_effects());
        assert_eq!(inst.dest(), None);
        assert_eq!(inst.operands(), vec![ValueId(0), ValueId(1)]);
    }

    #[test]
    fn test_barrier_has_side_effects() {
        let inst = MirInst::Barrier;
        assert!(inst.has_side_effects());
        assert_eq!(inst.dest(), None);
        assert_eq!(inst.operands(), [] as [ValueId; 0]);
    }
}
