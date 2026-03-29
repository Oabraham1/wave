// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Alias analysis for memory operations.
//!
//! Uses address space partitioning: local, device, and private memory
//! never alias each other. Within the same address space, uses offset
//! analysis to determine may-alias relationships.

use std::collections::{HashMap, HashSet};

use crate::hir::types::AddressSpace;
use crate::mir::function::MirFunction;
use crate::mir::instruction::MirInst;
use crate::mir::value::ValueId;

/// Alias query result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AliasResult {
    /// Definitely do not alias.
    NoAlias,
    /// Might alias (conservative).
    MayAlias,
    /// Definitely alias the same location.
    MustAlias,
}

/// Memory operation descriptor for alias queries.
#[derive(Debug, Clone)]
pub struct MemOp {
    /// Address value.
    pub addr: ValueId,
    /// Address space.
    pub space: AddressSpace,
}

/// Alias analysis results for a MIR function.
pub struct AliasInfo {
    /// Memory operations indexed by their instruction position.
    mem_ops: Vec<MemOp>,
    /// Map from value to its address space (if known).
    addr_spaces: HashMap<ValueId, AddressSpace>,
}

impl AliasInfo {
    /// Perform alias analysis on a MIR function.
    #[must_use]
    pub fn compute(func: &MirFunction) -> Self {
        let mut mem_ops = Vec::new();
        let mut addr_spaces: HashMap<ValueId, AddressSpace> = HashMap::new();

        for block in &func.blocks {
            for inst in &block.instructions {
                match inst {
                    MirInst::Load { addr, space, .. }
                    | MirInst::Store { addr, space, .. } => {
                        addr_spaces.insert(*addr, *space);
                        mem_ops.push(MemOp {
                            addr: *addr,
                            space: *space,
                        });
                    }
                    _ => {}
                }
            }
        }

        Self {
            mem_ops,
            addr_spaces,
        }
    }

    /// Query whether two memory operations may alias.
    #[must_use]
    pub fn query(&self, a: &MemOp, b: &MemOp) -> AliasResult {
        if a.space != b.space {
            return AliasResult::NoAlias;
        }
        if a.addr == b.addr {
            return AliasResult::MustAlias;
        }
        AliasResult::MayAlias
    }

    /// Returns all memory operations in the function.
    #[must_use]
    pub fn mem_ops(&self) -> &[MemOp] {
        &self.mem_ops
    }

    /// Returns the known address space of a value, if any.
    #[must_use]
    pub fn addr_space_of(&self, value: ValueId) -> Option<AddressSpace> {
        self.addr_spaces.get(&value).copied()
    }

    /// Returns all values that may alias a given memory operation.
    #[must_use]
    pub fn may_alias_set(&self, op: &MemOp) -> HashSet<ValueId> {
        let mut result = HashSet::new();
        for mop in &self.mem_ops {
            if self.query(op, mop) != AliasResult::NoAlias {
                result.insert(mop.addr);
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_different_address_spaces_no_alias() {
        let a = MemOp {
            addr: ValueId(0),
            space: AddressSpace::Device,
        };
        let b = MemOp {
            addr: ValueId(1),
            space: AddressSpace::Local,
        };
        let info = AliasInfo {
            mem_ops: vec![a.clone(), b.clone()],
            addr_spaces: HashMap::new(),
        };
        assert_eq!(info.query(&a, &b), AliasResult::NoAlias);
    }

    #[test]
    fn test_same_addr_must_alias() {
        let a = MemOp {
            addr: ValueId(0),
            space: AddressSpace::Device,
        };
        let b = MemOp {
            addr: ValueId(0),
            space: AddressSpace::Device,
        };
        let info = AliasInfo {
            mem_ops: vec![a.clone(), b.clone()],
            addr_spaces: HashMap::new(),
        };
        assert_eq!(info.query(&a, &b), AliasResult::MustAlias);
    }

    #[test]
    fn test_same_space_different_addr_may_alias() {
        let a = MemOp {
            addr: ValueId(0),
            space: AddressSpace::Device,
        };
        let b = MemOp {
            addr: ValueId(1),
            space: AddressSpace::Device,
        };
        let info = AliasInfo {
            mem_ops: vec![a.clone(), b.clone()],
            addr_spaces: HashMap::new(),
        };
        assert_eq!(info.query(&a, &b), AliasResult::MayAlias);
    }

    #[test]
    fn test_may_alias_set() {
        let ops = vec![
            MemOp {
                addr: ValueId(0),
                space: AddressSpace::Device,
            },
            MemOp {
                addr: ValueId(1),
                space: AddressSpace::Device,
            },
            MemOp {
                addr: ValueId(2),
                space: AddressSpace::Local,
            },
        ];
        let info = AliasInfo {
            mem_ops: ops.clone(),
            addr_spaces: HashMap::new(),
        };
        let aliases = info.may_alias_set(&ops[0]);
        assert!(aliases.contains(&ValueId(0)));
        assert!(aliases.contains(&ValueId(1)));
        assert!(!aliases.contains(&ValueId(2)));
    }
}
