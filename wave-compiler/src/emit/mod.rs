// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! WAVE instruction emission and binary generation.
//!
//! Converts LIR instructions with physical registers into encoded
//! WAVE machine code and packages it into the .wbin container format.

pub mod binary;
pub mod wave_emit;

pub use binary::{count_registers, generate_wbin};
pub use wave_emit::{emit_instruction, EncodedInst, RegMap};
