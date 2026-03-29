// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Low-Level Intermediate Representation (LIR) for WAVE kernels.
//!
//! LIR is close to WAVE machine instructions but uses virtual registers.
//! Each LIR instruction maps 1:1 to a WAVE ISA instruction. After
//! register allocation, virtual registers are replaced with physical ones.

pub mod display;
pub mod instruction;
pub mod operand;

pub use display::{display_lir, format_lir_inst};
pub use instruction::LirInst;
pub use operand::{MemWidth, PReg, PhysReg, SpecialReg, VReg};
