// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Escape analysis for stack-to-register promotion.
//!
//! Determines which locally-allocated values escape the current scope
//! (e.g., through stores to device memory or function calls). Non-escaping
//! values can be promoted from memory to registers.

use std::collections::HashSet;

use crate::mir::function::MirFunction;
use crate::mir::instruction::MirInst;
use crate::mir::value::ValueId;
use crate::hir::types::AddressSpace;

/// Result of escape analysis.
pub struct EscapeInfo {
    /// Values that escape the local scope.
    pub escaped: HashSet<ValueId>,
}

impl EscapeInfo {
    /// Compute escape information for a MIR function.
    #[must_use]
    pub fn compute(func: &MirFunction) -> Self {
        let mut escaped = HashSet::new();

        for block in &func.blocks {
            for inst in &block.instructions {
                match inst {
                    MirInst::Store {
                        value,
                        space: AddressSpace::Device,
                        ..
                    } => {
                        escaped.insert(*value);
                    }
                    MirInst::Call { args, .. } => {
                        for arg in args {
                            escaped.insert(*arg);
                        }
                    }
                    MirInst::AtomicRmw { value, .. } => {
                        escaped.insert(*value);
                    }
                    _ => {}
                }
            }
        }

        Self { escaped }
    }

    /// Returns true if a value escapes the local scope.
    #[must_use]
    pub fn escapes(&self, value: ValueId) -> bool {
        self.escaped.contains(&value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::expr::BinOp;
    use crate::mir::basic_block::{BasicBlock, Terminator};
    use crate::mir::instruction::{ConstValue, MirInst};
    use crate::mir::types::MirType;
    use crate::mir::value::BlockId;

    #[test]
    fn test_value_escapes_through_store() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb = BasicBlock::new(BlockId(0));
        bb.instructions.push(MirInst::Const {
            dest: ValueId(0),
            value: ConstValue::I32(42),
        });
        bb.instructions.push(MirInst::Store {
            addr: ValueId(1),
            value: ValueId(0),
            space: AddressSpace::Device,
        });
        bb.terminator = Terminator::Return;
        func.blocks.push(bb);

        let info = EscapeInfo::compute(&func);
        assert!(info.escapes(ValueId(0)));
        assert!(!info.escapes(ValueId(1)));
    }

    #[test]
    fn test_local_value_does_not_escape() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb = BasicBlock::new(BlockId(0));
        bb.instructions.push(MirInst::Const {
            dest: ValueId(0),
            value: ConstValue::I32(1),
        });
        bb.instructions.push(MirInst::Const {
            dest: ValueId(1),
            value: ConstValue::I32(2),
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

        let info = EscapeInfo::compute(&func);
        assert!(!info.escapes(ValueId(0)));
        assert!(!info.escapes(ValueId(1)));
        assert!(!info.escapes(ValueId(2)));
    }
}
