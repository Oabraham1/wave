// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Optimization pass trait definition.
//!
//! All optimization passes implement this trait, allowing them to be
//! composed in a pipeline and run until a fixed point is reached.

use crate::mir::function::MirFunction;

/// Trait for MIR optimization passes.
pub trait Pass {
    /// Returns the name of this pass.
    fn name(&self) -> &'static str;

    /// Run the pass on a MIR function. Returns true if any changes were made.
    fn run(&self, func: &mut MirFunction) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::value::BlockId;

    struct NoopPass;
    impl Pass for NoopPass {
        fn name(&self) -> &'static str {
            "noop"
        }
        fn run(&self, _func: &mut MirFunction) -> bool {
            false
        }
    }

    #[test]
    fn test_pass_trait() {
        let pass = NoopPass;
        assert_eq!(pass.name(), "noop");
    }

    #[test]
    fn test_pass_returns_false_when_no_changes() {
        let pass = NoopPass;
        let mut func = MirFunction::new("test".into(), BlockId(0));
        assert!(!pass.run(&mut func));
    }
}
