// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Basic blocks with terminators for MIR control flow graphs.
//!
//! A basic block is a straight-line sequence of instructions ending
//! with a terminator that transfers control to one or more successor blocks.

use super::instruction::MirInst;
use super::value::{BlockId, ValueId};
use crate::mir::types::MirType;

/// Phi node at a block join point, merging values from different predecessors.
#[derive(Debug, Clone, PartialEq)]
pub struct PhiNode {
    /// Destination value.
    pub dest: ValueId,
    /// Type of the phi value.
    pub ty: MirType,
    /// Incoming values from predecessor blocks.
    pub incoming: Vec<(BlockId, ValueId)>,
}

/// Block terminator transferring control to successor blocks.
#[derive(Debug, Clone, PartialEq)]
pub enum Terminator {
    /// Unconditional branch.
    Branch {
        /// Target block.
        target: BlockId,
    },
    /// Conditional branch.
    CondBranch {
        /// Condition (must be Bool type).
        cond: ValueId,
        /// Target if condition is true.
        true_target: BlockId,
        /// Target if condition is false.
        false_target: BlockId,
    },
    /// Return from kernel.
    Return,
}

impl Terminator {
    /// Returns all successor block IDs.
    #[must_use]
    pub fn successors(&self) -> Vec<BlockId> {
        match self {
            Self::Branch { target } => vec![*target],
            Self::CondBranch {
                true_target,
                false_target,
                ..
            } => vec![*true_target, *false_target],
            Self::Return => vec![],
        }
    }

    /// Returns all value IDs used by this terminator.
    #[must_use]
    pub fn operands(&self) -> Vec<ValueId> {
        match self {
            Self::Branch { .. } | Self::Return => vec![],
            Self::CondBranch { cond, .. } => vec![*cond],
        }
    }
}

/// A basic block in the MIR control flow graph.
#[derive(Debug, Clone, PartialEq)]
pub struct BasicBlock {
    /// Unique block identifier.
    pub id: BlockId,
    /// Phi nodes at the start of this block.
    pub phis: Vec<PhiNode>,
    /// Instructions in this block.
    pub instructions: Vec<MirInst>,
    /// Block terminator.
    pub terminator: Terminator,
}

impl BasicBlock {
    /// Create a new empty basic block.
    #[must_use]
    pub fn new(id: BlockId) -> Self {
        Self {
            id,
            phis: Vec::new(),
            instructions: Vec::new(),
            terminator: Terminator::Return,
        }
    }

    /// Returns all successor block IDs.
    #[must_use]
    pub fn successors(&self) -> Vec<BlockId> {
        self.terminator.successors()
    }

    /// Returns all value IDs defined in this block (phi dests + instruction dests).
    #[must_use]
    pub fn defs(&self) -> Vec<ValueId> {
        let mut defs: Vec<ValueId> = self.phis.iter().map(|phi| phi.dest).collect();
        for inst in &self.instructions {
            if let Some(dest) = inst.dest() {
                defs.push(dest);
            }
        }
        defs
    }

    /// Returns all value IDs used in this block.
    #[must_use]
    pub fn uses(&self) -> Vec<ValueId> {
        let mut uses = Vec::new();
        for phi in &self.phis {
            for (_, val) in &phi.incoming {
                uses.push(*val);
            }
        }
        for inst in &self.instructions {
            uses.extend(inst.operands());
        }
        uses.extend(self.terminator.operands());
        uses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::expr::BinOp;
    use crate::mir::types::MirType;

    #[test]
    fn test_basic_block_successors() {
        let mut bb = BasicBlock::new(BlockId(0));
        bb.terminator = Terminator::CondBranch {
            cond: ValueId(0),
            true_target: BlockId(1),
            false_target: BlockId(2),
        };
        let succs = bb.successors();
        assert_eq!(succs, vec![BlockId(1), BlockId(2)]);
    }

    #[test]
    fn test_basic_block_defs_and_uses() {
        let mut bb = BasicBlock::new(BlockId(0));
        bb.instructions.push(MirInst::BinOp {
            dest: ValueId(2),
            op: BinOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: MirType::I32,
        });
        bb.terminator = Terminator::Return;

        let defs = bb.defs();
        assert_eq!(defs, vec![ValueId(2)]);

        let uses = bb.uses();
        assert_eq!(uses, vec![ValueId(0), ValueId(1)]);
    }

    #[test]
    fn test_phi_node() {
        let phi = PhiNode {
            dest: ValueId(3),
            ty: MirType::I32,
            incoming: vec![(BlockId(0), ValueId(1)), (BlockId(1), ValueId(2))],
        };
        assert_eq!(phi.dest, ValueId(3));
        assert_eq!(phi.incoming.len(), 2);
    }

    #[test]
    fn test_terminator_operands() {
        let term = Terminator::CondBranch {
            cond: ValueId(5),
            true_target: BlockId(1),
            false_target: BlockId(2),
        };
        assert_eq!(term.operands(), vec![ValueId(5)]);

        let ret = Terminator::Return;
        assert!(ret.operands().is_empty());
    }
}
