// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Live range computation for virtual registers.
//!
//! Computes the program points where each virtual register is live,
//! represented as intervals over instruction indices in the flat LIR.

use std::collections::{HashMap, HashSet};

use crate::lir::instruction::LirInst;
use crate::lir::operand::VReg;

/// A live range interval [start, end) for a virtual register.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveRange {
    /// First instruction index where the value is defined.
    pub start: usize,
    /// Last instruction index where the value is used (inclusive).
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

/// Compute live ranges for all virtual registers in a LIR instruction sequence.
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

        for src in inst.src_vregs() {
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

    ranges
}

/// Collect all virtual registers used in the instruction sequence.
#[must_use]
pub fn collect_vregs(instructions: &[LirInst]) -> HashSet<VReg> {
    let mut vregs = HashSet::new();
    for inst in instructions {
        if let Some(dest) = inst.dest_vreg() {
            vregs.insert(dest);
        }
        for src in inst.src_vregs() {
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
