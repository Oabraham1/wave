// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Analysis passes for MIR: CFG, dominance, liveness, alias, escape, and loop analysis.
//!
//! Analysis passes compute information used by optimization passes and
//! the register allocator. They do not modify the IR.

pub mod alias;
pub mod cfg;
pub mod dominance;
pub mod escape;
pub mod liveness;
pub mod loop_analysis;

pub use alias::{AliasInfo, AliasResult};
pub use cfg::Cfg;
pub use dominance::DomTree;
pub use escape::EscapeInfo;
pub use liveness::LivenessInfo;
pub use loop_analysis::LoopInfo;
