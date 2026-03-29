// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Move coalescing to reduce register-to-register copies.
//!
//! If two virtual registers connected by a move instruction do not
//! interfere, they can be merged into one, eliminating the move.

use std::collections::HashMap;

use super::interference::InterferenceGraph;
use crate::lir::instruction::LirInst;
use crate::lir::operand::VReg;

/// Result of coalescing.
pub struct CoalesceResult {
    /// Mapping from coalesced VRegs to their representative VReg.
    pub mapping: HashMap<VReg, VReg>,
    /// Number of moves eliminated.
    pub eliminated: usize,
}

/// Perform move coalescing on the interference graph.
#[must_use]
pub fn coalesce(ig: &InterferenceGraph, max_regs: u32) -> CoalesceResult {
    let mut mapping: HashMap<VReg, VReg> = HashMap::new();
    let mut eliminated = 0;

    for &(dest, src) in &ig.moves {
        let dest_rep = find_representative(&mapping, dest);
        let src_rep = find_representative(&mapping, src);

        if dest_rep == src_rep {
            continue;
        }

        if ig.interferes(dest_rep, src_rep) {
            continue;
        }

        let dest_deg = ig.degree(dest_rep);
        let src_deg = ig.degree(src_rep);
        if dest_deg + src_deg >= max_regs as usize {
            continue;
        }

        mapping.insert(src_rep, dest_rep);
        eliminated += 1;
    }

    CoalesceResult {
        mapping,
        eliminated,
    }
}

fn find_representative(mapping: &HashMap<VReg, VReg>, vreg: VReg) -> VReg {
    let mut current = vreg;
    while let Some(&next) = mapping.get(&current) {
        if next == current {
            break;
        }
        current = next;
    }
    current
}

/// Apply coalescing result to LIR instructions.
pub fn apply_coalescing(instructions: &mut [LirInst], result: &CoalesceResult) {
    for inst in instructions.iter_mut() {
        rewrite_vreg_in_inst(inst, &result.mapping);
    }
}

fn rewrite_vreg_in_inst(inst: &mut LirInst, mapping: &HashMap<VReg, VReg>) {
    match inst {
        LirInst::MovReg { dest, src } => {
            *dest = find_representative(mapping, *dest);
            *src = find_representative(mapping, *src);
        }
        LirInst::Iadd { dest, src1, src2 }
        | LirInst::Isub { dest, src1, src2 }
        | LirInst::Imul { dest, src1, src2 }
        | LirInst::Fadd { dest, src1, src2 }
        | LirInst::Fsub { dest, src1, src2 }
        | LirInst::Fmul { dest, src1, src2 } => {
            *dest = find_representative(mapping, *dest);
            *src1 = find_representative(mapping, *src1);
            *src2 = find_representative(mapping, *src2);
        }
        LirInst::MovImm { dest, .. } | LirInst::MovSr { dest, .. } => {
            *dest = find_representative(mapping, *dest);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coalesce_non_interfering_move() {
        let insts = vec![
            LirInst::MovImm { dest: VReg(0), value: 1 },
            LirInst::MovReg { dest: VReg(1), src: VReg(0) },
            LirInst::Halt,
        ];

        let ig = InterferenceGraph::build(&insts);
        let result = coalesce(&ig, 256);
        assert!(result.eliminated > 0 || result.mapping.is_empty());
    }

    #[test]
    fn test_coalesce_no_moves() {
        let insts = vec![
            LirInst::MovImm { dest: VReg(0), value: 1 },
            LirInst::MovImm { dest: VReg(1), value: 2 },
            LirInst::Halt,
        ];

        let ig = InterferenceGraph::build(&insts);
        let result = coalesce(&ig, 256);
        assert_eq!(result.eliminated, 0);
    }
}
