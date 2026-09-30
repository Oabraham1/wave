// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! WAVE instructions with virtual registers. Register allocation assigns physical registers
//! before emission.

pub mod display;
pub mod instruction;
pub mod operand;

pub use display::{display_lir, format_lir_inst};
pub use instruction::LirInst;
pub use operand::{MemWidth, PReg, PhysReg, SpecialReg, VReg};
