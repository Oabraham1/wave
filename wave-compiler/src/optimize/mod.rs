// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Optimization pipeline for MIR.
//!
//! Includes dead code elimination, common subexpression elimination,
//! constant folding, loop-invariant code motion, strength reduction,
//! SCCP, mem2reg, loop unrolling, and CFG simplification.
//! Passes are run in a fixed order until no pass reports changes.

pub mod constant_fold;
pub mod cse;
pub mod dce;
pub mod licm;
pub mod loop_unroll;
pub mod mem2reg;
pub mod pass;
pub mod sccp;
pub mod simplify_cfg;
pub mod strength_reduce;

use crate::driver::config::OptLevel;
use crate::mir::function::MirFunction;
use pass::Pass;

/// Run the optimization pipeline on a MIR function at the given optimization level.
pub fn optimize(func: &mut MirFunction, opt_level: OptLevel) {
    let passes: Vec<Box<dyn Pass>> = match opt_level {
        OptLevel::O0 => vec![],
        OptLevel::O1 => vec![
            Box::new(mem2reg::Mem2Reg),
            Box::new(constant_fold::ConstantFold),
            Box::new(dce::Dce),
            Box::new(simplify_cfg::SimplifyCfg),
        ],
        OptLevel::O2 => vec![
            Box::new(mem2reg::Mem2Reg),
            Box::new(sccp::Sccp),
            Box::new(dce::Dce),
            Box::new(cse::Cse),
            Box::new(licm::Licm),
            Box::new(strength_reduce::StrengthReduce),
            Box::new(simplify_cfg::SimplifyCfg),
            Box::new(dce::Dce),
        ],
        OptLevel::O3 => vec![
            Box::new(mem2reg::Mem2Reg),
            Box::new(sccp::Sccp),
            Box::new(dce::Dce),
            Box::new(cse::Cse),
            Box::new(licm::Licm),
            Box::new(loop_unroll::LoopUnroll),
            Box::new(strength_reduce::StrengthReduce),
            Box::new(simplify_cfg::SimplifyCfg),
            Box::new(dce::Dce),
        ],
    };

    let mut changed = true;
    let mut iterations = 0;
    while changed && iterations < 10 {
        changed = false;
        for pass in &passes {
            changed |= pass.run(func);
        }
        iterations += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::basic_block::{BasicBlock, Terminator};
    use crate::mir::instruction::{ConstValue, MirInst};
    use crate::mir::value::{BlockId, ValueId};

    #[test]
    fn test_optimize_o0_no_changes() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb = BasicBlock::new(BlockId(0));
        bb.instructions.push(MirInst::Const {
            dest: ValueId(0),
            value: ConstValue::I32(42),
        });
        bb.terminator = Terminator::Return;
        func.blocks.push(bb);

        optimize(&mut func, OptLevel::O0);
        assert_eq!(func.blocks[0].instructions.len(), 1);
    }

    #[test]
    fn test_optimize_o1_removes_dead_code() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb = BasicBlock::new(BlockId(0));
        bb.instructions.push(MirInst::Const {
            dest: ValueId(0),
            value: ConstValue::I32(42),
        });
        bb.instructions.push(MirInst::Const {
            dest: ValueId(1),
            value: ConstValue::I32(99),
        });
        bb.terminator = Terminator::Return;
        func.blocks.push(bb);

        optimize(&mut func, OptLevel::O1);
        assert!(func.blocks[0].instructions.is_empty());
    }
}
