// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Connect virtual registers with overlapping live ranges. Parameter ranges start at function
//! entry.

use std::collections::{HashMap, HashSet};

use super::live_range::{compute_live_ranges, LiveRange};
use crate::lir::instruction::LirInst;
use crate::lir::operand::VReg;

#[derive(Debug)]
/// Interference graph for register allocation.
pub struct InterferenceGraph {
    /// Adjacency sets: for each `VReg`, the set of interfering `VReg`s.
    pub adj: HashMap<VReg, HashSet<VReg>>,
    /// Move pairs, dest, src, for coalescing.
    pub moves: Vec<(VReg, VReg)>,
    /// Live ranges for each `VReg`.
    pub ranges: HashMap<VReg, LiveRange>,
}

impl InterferenceGraph {
    /// Build the interference graph from LIR instructions.
    #[must_use]
    pub fn build(instructions: &[LirInst]) -> Self {
        Self::build_with_params(instructions, 0)
    }

    /// Build the interference graph, extending param `VReg` live ranges to start at 0.
    #[must_use]
    pub fn build_with_params(instructions: &[LirInst], num_params: u32) -> Self {
        let mut ranges = compute_live_ranges(instructions);

        for i in 0..num_params {
            let vreg = VReg(i);
            ranges
                .entry(vreg)
                .and_modify(|r| r.start = 0)
                .or_insert(LiveRange { start: 0, end: 0 });
        }

        let mut adj: HashMap<VReg, HashSet<VReg>> = HashMap::new();
        let mut moves = Vec::new();

        let mut intervals: Vec<_> = ranges.iter().map(|(&vreg, range)| (vreg, range)).collect();
        intervals.sort_unstable_by_key(|(vreg, range)| (range.start, *vreg));
        let mut active: Vec<(VReg, &LiveRange)> = Vec::new();
        for (vreg, range) in intervals {
            active.retain(|(_, other)| other.end >= range.start);
            let mut neighbors = HashSet::with_capacity(active.len());
            for &(other, _) in &active {
                neighbors.insert(other);
                adj.entry(other).or_default().insert(vreg);
            }
            adj.insert(vreg, neighbors);
            active.push((vreg, range));
        }

        for inst in instructions {
            if let LirInst::MovReg { dest, src } = inst {
                if !adj.get(dest).is_some_and(|s| s.contains(src)) {
                    moves.push((*dest, *src));
                }
            }
        }

        Self { adj, moves, ranges }
    }

    /// Returns the degree, number of neighbors, of a `VReg`.
    #[must_use]
    pub fn degree(&self, vreg: VReg) -> usize {
        self.adj.get(&vreg).map_or(0, HashSet::len)
    }

    /// Returns true if two `VReg`s interfere.
    #[must_use]
    pub fn interferes(&self, a: VReg, b: VReg) -> bool {
        self.adj.get(&a).is_some_and(|s| s.contains(&b))
    }

    /// Returns all `VReg`s in the graph.
    #[must_use]
    pub fn nodes(&self) -> Vec<VReg> {
        self.adj.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interference_graph_simple() {
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
        assert!(ig.interferes(VReg(0), VReg(1)));
        assert!(ig.degree(VReg(0)) >= 1);
    }

    #[test]
    fn test_non_overlapping_no_interference() {
        let insts = vec![
            LirInst::MovImm {
                dest: VReg(0),
                value: 1,
            },
            LirInst::MovImm {
                dest: VReg(1),
                value: 2,
            },
        ];

        let ig = InterferenceGraph::build(&insts);
        assert!(!ig.interferes(VReg(0), VReg(1)));
    }

    #[test]
    fn test_move_between_interfering_not_coalescable() {
        let insts = vec![
            LirInst::MovImm {
                dest: VReg(0),
                value: 1,
            },
            LirInst::MovReg {
                dest: VReg(1),
                src: VReg(0),
            },
            LirInst::Halt,
        ];

        let ig = InterferenceGraph::build(&insts);
        assert_eq!(ig.moves, [] as [(VReg, VReg); 0]);
    }

    #[test]
    fn test_params_interfere_with_later_temps() {
        let insts = vec![
            LirInst::MovImm {
                dest: VReg(2),
                value: 4,
            },
            LirInst::Iadd {
                dest: VReg(3),
                src1: VReg(0),
                src2: VReg(2),
            },
            LirInst::Iadd {
                dest: VReg(4),
                src1: VReg(1),
                src2: VReg(2),
            },
            LirInst::Halt,
        ];

        let ig = InterferenceGraph::build_with_params(&insts, 2);
        assert!(ig.interferes(VReg(0), VReg(2)));
        assert!(ig.interferes(VReg(1), VReg(2)));
    }
    #[test]
    fn interference_matches_inclusive_live_ranges() {
        let mut instructions = Vec::new();
        for index in 0..128 {
            instructions.push(LirInst::MovImm {
                dest: VReg(index + 4),
                value: index,
            });
            instructions.push(LirInst::Iadd {
                dest: VReg(index + 132),
                src1: VReg(index + 4),
                src2: VReg(index % 4),
            });
        }
        let graph = InterferenceGraph::build_with_params(&instructions, 4);
        for (&a, range_a) in &graph.ranges {
            for (&b, range_b) in &graph.ranges {
                assert_eq!(
                    graph.interferes(a, b),
                    a != b && range_a.overlaps(range_b),
                    "{a:?}, {b:?}"
                );
            }
        }
    }
}
