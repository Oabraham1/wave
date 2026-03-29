// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Pretty-printing for MIR functions and instructions.
//!
//! Provides human-readable output of the MIR for debugging and
//! compiler development purposes.

use std::fmt;

use super::basic_block::Terminator;
use super::function::MirFunction;
use super::instruction::MirInst;

/// Format a MIR function for display.
pub fn display_function(func: &MirFunction) -> String {
    let mut out = String::new();
    out.push_str(&format!("function {} ({} params):\n", func.name, func.params.len()));
    for param in &func.params {
        out.push_str(&format!("  param {} : {} ({})\n", param.value, param.ty, param.name));
    }
    out.push('\n');
    for block in &func.blocks {
        out.push_str(&format!("{}:\n", block.id));
        for phi in &block.phis {
            out.push_str(&format!("  {} = phi {}", phi.dest, phi.ty));
            for (bid, val) in &phi.incoming {
                out.push_str(&format!(" [{}: {}]", bid, val));
            }
            out.push('\n');
        }
        for inst in &block.instructions {
            out.push_str(&format!("  {}\n", format_inst(inst)));
        }
        out.push_str(&format!("  {}\n", format_terminator(&block.terminator)));
    }
    out
}

fn format_inst(inst: &MirInst) -> String {
    match inst {
        MirInst::BinOp {
            dest,
            op,
            lhs,
            rhs,
            ty,
        } => format!("{dest} = {op:?} {ty} {lhs}, {rhs}"),
        MirInst::UnaryOp {
            dest,
            op,
            operand,
            ty,
        } => format!("{dest} = {op:?} {ty} {operand}"),
        MirInst::Load {
            dest,
            addr,
            space,
            ty,
        } => format!("{dest} = load {ty} {space} [{addr}]"),
        MirInst::Store {
            addr,
            value,
            space,
        } => format!("store {space} [{addr}], {value}"),
        MirInst::Call { dest, func, args } => {
            let args_str: Vec<String> = args.iter().map(ToString::to_string).collect();
            match dest {
                Some(d) => format!("{d} = call {func:?}({args})", args = args_str.join(", ")),
                None => format!("call {func:?}({args})", args = args_str.join(", ")),
            }
        }
        MirInst::Cast {
            dest,
            value,
            from,
            to,
        } => format!("{dest} = cast {from} -> {to} {value}"),
        MirInst::Const { dest, value } => format!("{dest} = const {value:?}"),
        MirInst::Shuffle {
            dest,
            value,
            lane,
            mode,
        } => format!("{dest} = shuffle {mode:?} {value}, {lane}"),
        MirInst::AtomicRmw {
            dest,
            addr,
            value,
            op,
            ..
        } => format!("{dest} = atomic_{op:?} [{addr}], {value}"),
        MirInst::ReadSpecialReg { dest, sr_index } => format!("{dest} = read_sr {sr_index}"),
        MirInst::Barrier => "barrier".to_string(),
        MirInst::Fence { scope } => format!("fence {scope:?}"),
    }
}

fn format_terminator(term: &Terminator) -> String {
    match term {
        Terminator::Branch { target } => format!("br {target}"),
        Terminator::CondBranch {
            cond,
            true_target,
            false_target,
        } => format!("br {cond}, {true_target}, {false_target}"),
        Terminator::Return => "ret".to_string(),
    }
}

impl fmt::Display for MirFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", display_function(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::expr::BinOp;
    use crate::mir::basic_block::{BasicBlock, Terminator};
    use crate::mir::instruction::{ConstValue, MirInst};
    use crate::mir::types::MirType;
    use crate::mir::value::{BlockId, ValueId};

    #[test]
    fn test_display_simple_function() {
        let mut func = MirFunction::new("add".into(), BlockId(0));
        let mut bb = BasicBlock::new(BlockId(0));
        bb.instructions.push(MirInst::Const {
            dest: ValueId(0),
            value: ConstValue::I32(42),
        });
        bb.terminator = Terminator::Return;
        func.blocks.push(bb);

        let output = display_function(&func);
        assert!(output.contains("function add"));
        assert!(output.contains("const"));
        assert!(output.contains("ret"));
    }

    #[test]
    fn test_display_branch() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        let mut bb = BasicBlock::new(BlockId(0));
        bb.terminator = Terminator::CondBranch {
            cond: ValueId(0),
            true_target: BlockId(1),
            false_target: BlockId(2),
        };
        func.blocks.push(bb);

        let output = display_function(&func);
        assert!(output.contains("br %0, bb1, bb2"));
    }

    #[test]
    fn test_display_binop() {
        let inst = MirInst::BinOp {
            dest: ValueId(2),
            op: BinOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: MirType::I32,
        };
        let s = format_inst(&inst);
        assert!(s.contains("%2 = "));
        assert!(s.contains("Add"));
        assert!(s.contains("%0"));
        assert!(s.contains("%1"));
    }
}
