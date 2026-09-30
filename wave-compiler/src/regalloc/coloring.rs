// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Assign physical registers by coloring the interference graph. Parameter registers keep their
//! argument indices.

use std::collections::HashMap;

use super::interference::InterferenceGraph;
use crate::lir::operand::{PhysReg, VReg};

#[derive(Debug)]
/// Result of graph coloring.
pub struct ColoringResult {
    /// Mapping from `VReg` to assigned `PhysReg`.
    pub assignment: HashMap<VReg, PhysReg>,
    /// `VReg`s that could not be colored and need spilling.
    pub spilled: Vec<VReg>,
}

/// Perform graph coloring with pre-colored parameter registers.
///
/// Parameters VReg(0)..VReg(num_params-1) are pre-colored to
/// PhysReg(0)..PhysReg(num_params-1) per Rule 9.
#[must_use]
pub fn color(ig: &InterferenceGraph, max_regs: u32, num_params: u32) -> ColoringResult {
    let mut assignment: HashMap<VReg, PhysReg> = HashMap::with_capacity(ig.adj.len());
    let mut spilled = Vec::new();

    for physical in (0..=u8::MAX).take(num_params.min(max_regs).min(256) as usize) {
        let vreg = VReg(u32::from(physical));
        if ig.adj.contains_key(&vreg) {
            assignment.insert(vreg, PhysReg(physical));
        }
    }

    let mut nodes: Vec<VReg> = ig.nodes();
    nodes.sort_unstable_by_key(|v| (std::cmp::Reverse(ig.degree(*v)), *v));

    let mut stack = Vec::with_capacity(nodes.len());

    for &vreg in &nodes {
        if !assignment.contains_key(&vreg) {
            stack.push(vreg);
        }
    }

    while let Some(vreg) = stack.pop() {
        let mut used_colors = [false; 256];
        if let Some(neighbors) = ig.adj.get(&vreg) {
            for &neighbor in neighbors {
                if let Some(&phys) = assignment.get(&neighbor) {
                    used_colors[usize::from(phys.0)] = true;
                }
            }
        }

        if let Some(available) = (0..=u8::MAX)
            .take(max_regs.min(256) as usize)
            .find(|color| !used_colors[usize::from(*color)])
        {
            assignment.insert(vreg, PhysReg(available));
        } else {
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
            LirInst::MovImm {
                dest: VReg(0),
                value: 1,
            },
            LirInst::MovImm {
                dest: VReg(1),
                value: 2,
            },
            LirInst::Iadd {
                dest: VReg(2),
                src1: VReg(0),
                src2: VReg(1),
            },
            LirInst::Halt,
        ];

        let ig = InterferenceGraph::build(&insts);
        let result = color(&ig, 32, 0);

        assert_eq!(result.spilled, [] as [VReg; 0]);
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
