// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Graph coloring register allocator following Chaitin-Briggs.
//!
//! Assigns physical registers to virtual registers by coloring the
//! interference graph. Pre-colors parameter registers per Rule 9:
//! VReg(0)→PhysReg(0), VReg(1)→PhysReg(1), etc.

use std::collections::{HashMap, HashSet};

use super::interference::InterferenceGraph;
use crate::lir::operand::{PhysReg, VReg};

/// Result of graph coloring.
pub struct ColoringResult {
    /// Mapping from VReg to assigned PhysReg.
    pub assignment: HashMap<VReg, PhysReg>,
    /// VRegs that could not be colored and need spilling.
    pub spilled: Vec<VReg>,
}

/// Perform graph coloring with pre-colored parameter registers.
///
/// Parameters VReg(0)..VReg(num_params-1) are pre-colored to
/// PhysReg(0)..PhysReg(num_params-1) per Rule 9.
#[must_use]
pub fn color(ig: &InterferenceGraph, max_regs: u32, num_params: u32) -> ColoringResult {
    let mut assignment: HashMap<VReg, PhysReg> = HashMap::new();
    let mut spilled = Vec::new();

    for i in 0..num_params {
        let vreg = VReg(i);
        if ig.adj.contains_key(&vreg) {
            assignment.insert(vreg, PhysReg(i as u8));
        }
    }

    let mut nodes: Vec<VReg> = ig.nodes();
    nodes.sort_by_key(|v| std::cmp::Reverse(ig.degree(*v)));

    let mut stack: Vec<VReg> = Vec::new();
    let mut removed: HashSet<VReg> = HashSet::new();

    for &vreg in &nodes {
        if !assignment.contains_key(&vreg) {
            stack.push(vreg);
            removed.insert(vreg);
        }
    }

    while let Some(vreg) = stack.pop() {
        let mut used_colors: HashSet<u8> = HashSet::new();
        if let Some(neighbors) = ig.adj.get(&vreg) {
            for &neighbor in neighbors {
                if let Some(&phys) = assignment.get(&neighbor) {
                    used_colors.insert(phys.0);
                }
            }
        }

        let mut color_found = false;
        for c in 0..max_regs.min(32) as u8 {
            if !used_colors.contains(&c) {
                assignment.insert(vreg, PhysReg(c));
                color_found = true;
                break;
            }
        }

        if !color_found {
            spilled.push(vreg);
        }
    }

    ColoringResult {
        assignment,
        spilled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lir::instruction::LirInst;

    #[test]
    fn test_coloring_simple() {
        let insts = vec![
            LirInst::MovImm { dest: VReg(0), value: 1 },
            LirInst::MovImm { dest: VReg(1), value: 2 },
            LirInst::Iadd {
                dest: VReg(2),
                src1: VReg(0),
                src2: VReg(1),
            },
            LirInst::Halt,
        ];

        let ig = InterferenceGraph::build(&insts);
        let result = color(&ig, 32, 0);

        assert!(result.spilled.is_empty());
        assert!(result.assignment.contains_key(&VReg(0)));
        assert!(result.assignment.contains_key(&VReg(1)));
        assert!(result.assignment.contains_key(&VReg(2)));

        let r0 = result.assignment[&VReg(0)];
        let r1 = result.assignment[&VReg(1)];
        if ig.interferes(VReg(0), VReg(1)) {
            assert_ne!(r0, r1);
        }
    }

    #[test]
    fn test_coloring_precolors_params() {
        let insts = vec![
            LirInst::Iadd {
                dest: VReg(2),
                src1: VReg(0),
                src2: VReg(1),
            },
            LirInst::Halt,
        ];

        let ig = InterferenceGraph::build(&insts);
        let result = color(&ig, 32, 2);

        assert_eq!(result.assignment[&VReg(0)], PhysReg(0));
        assert_eq!(result.assignment[&VReg(1)], PhysReg(1));
    }
}
