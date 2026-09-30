// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Typed SSA representation with basic blocks and phi nodes for analysis and optimization.

pub mod basic_block;
pub mod builder;
pub mod display;
pub mod function;
pub mod instruction;
pub mod types;
pub mod value;

pub use basic_block::{BasicBlock, PhiNode, Terminator};
pub use builder::MirBuilder;
pub use function::{MirFunction, MirParam};
pub use instruction::{AtomicOp, ConstValue, MirInst};
pub use types::MirType;
pub use value::{BlockId, IdGenerator, ValueId};
