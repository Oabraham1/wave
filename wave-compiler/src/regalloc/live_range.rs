// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Live range computation for virtual registers. Computes the program points where each virtual
//! register is live represented as intervals over instruction indices in the flat LIR.

use std::collections::{HashMap, HashSet};

use crate::lir::instruction::LirInst;
use crate::lir::operand::VReg;

/// An inclusive live range interval for a virtual register.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveRange {
    /// First instruction index covered by the interval.
    pub start: usize,
    /// Last instruction index covered by the interval.
    pub end: usize,
}

impl LiveRange {
    /// Returns true if this live range overlaps with another.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    /// Returns the length of this live range.
    #[must_use]
    pub fn length(&self) -> usize {
        if self.end >= self.start {
            self.end - self.start + 1
        } else {
            0
        }
    }
}

/// Compute inclusive live ranges, preserving values needed on loop back edges.
///
/// # Examples
///
/// ```
/// use wave_compiler::lir::{instruction::LirInst, operand::VReg};
/// use wave_compiler::regalloc::live_range::{compute_live_ranges, LiveRange};
///
/// let instructions = [LirInst::MovImm { dest: VReg(0), value: 7 }];
/// let ranges = compute_live_ranges(&instructions);
/// assert_eq!(ranges[&VReg(0)], LiveRange { start: 0, end: 0 });
/// ```
#[must_use]
pub fn compute_live_ranges(instructions: &[LirInst]) -> HashMap<VReg, LiveRange> {
    let mut ranges: HashMap<VReg, LiveRange> = HashMap::new();

    for (idx, inst) in instructions.iter().enumerate() {
        if let Some(dest) = inst.dest_vreg() {
            ranges
                .entry(dest)
                .and_modify(|r| {
                    if idx < r.start {
                        r.start = idx;
                    }
                    if idx > r.end {
                        r.end = idx;
                    }
                })
                .or_insert(LiveRange {
                    start: idx,
                    end: idx,
                });
        }

        for src in inst.src_vreg_iter() {
            ranges
                .entry(src)
                .and_modify(|r| {
                    if idx < r.start {
                        r.start = idx;
                    }
                    if idx > r.end {
                        r.end = idx;
                    }
                })
                .or_insert(LiveRange {
                    start: idx,
                    end: idx,
                });
        }
    }

    let mut loops = Vec::new();
    for (index, instruction) in instructions.iter().enumerate() {
        match instruction {
            LirInst::Loop => loops.push(index),
            LirInst::Endloop => {
                if let Some(start) = loops.pop() {
                    extend_loop_inputs(instructions, &mut ranges, start, index);
                }
            }
            _ => {}
        }
    }

    ranges
}

fn extend_loop_inputs(
    instructions: &[LirInst],
    ranges: &mut HashMap<VReg, LiveRange>,
    start: usize,
    end: usize,
) {
    let mut defined = HashSet::new();
    let mut scopes = Vec::new();
    for instruction in &instructions[start + 1..end] {
        for source in instruction.src_vreg_iter() {
            if !defined.contains(&source) {
                // A read before a dominating definition needs the value on every back edge.
                if let Some(range) = ranges.get_mut(&source) {
                    range.start = range.start.min(start);
                    range.end = range.end.max(end);
                }
            }
        }
        if let Some(destination) = instruction.dest_vreg() {
            defined.insert(destination);
        }
        match instruction {
            LirInst::If { .. } | LirInst::Loop => scopes.push(defined.clone()),
            LirInst::Else => {
                if let Some(entry) = scopes.last() {
                    defined.clone_from(entry);
                }
            }
            LirInst::Endif | LirInst::Endloop => {
                if let Some(entry) = scopes.pop() {
                    defined = entry;
                }
            }
            _ => {}
        }
    }
}

/// Collect all virtual registers used in the instruction sequence.
#[must_use]
pub fn collect_vregs(instructions: &[LirInst]) -> HashSet<VReg> {
    let mut vregs = HashSet::new();
    for inst in instructions {
        if let Some(dest) = inst.dest_vreg() {
            vregs.insert(dest);
        }
        for src in inst.src_vreg_iter() {
            vregs.insert(src);
        }
    }
    vregs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_live_range_overlap() {
        let a = LiveRange { start: 0, end: 5 };
        let b = LiveRange { start: 3, end: 8 };
        let c = LiveRange { start: 6, end: 10 };
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
        assert!(b.overlaps(&c));
    }

    #[test]
    fn test_compute_live_ranges() {
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

        let ranges = compute_live_ranges(&insts);
        assert_eq!(ranges[&VReg(0)], LiveRange { start: 0, end: 2 });
        assert_eq!(ranges[&VReg(1)], LiveRange { start: 1, end: 2 });
        assert_eq!(ranges[&VReg(2)], LiveRange { start: 2, end: 2 });
    }

    #[test]
    fn test_collect_vregs() {
        let insts = vec![
            LirInst::Iadd {
                dest: VReg(2),
                src1: VReg(0),
                src2: VReg(1),
            },
            LirInst::Halt,
        ];
        let vregs = collect_vregs(&insts);
        assert_eq!(vregs.len(), 3);
        assert!(vregs.contains(&VReg(0)));
        assert!(vregs.contains(&VReg(1)));
        assert!(vregs.contains(&VReg(2)));
    }
}
