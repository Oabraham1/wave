// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Register allocator using graph coloring (Chaitin-Briggs).
//!
//! Assigns physical registers to virtual registers. Handles interference
//! graph construction, move coalescing, graph coloring with pre-colored
//! parameters, and spill code generation. After allocation, rewrites
//! VRegs to PhysRegs in the instruction stream.

pub mod coalesce;
pub mod coloring;
pub mod interference;
pub mod live_range;
pub mod spill;

use std::collections::HashMap;

use crate::emit::wave_emit::RegMap;
use crate::lir::instruction::LirInst;
use crate::lir::operand::VReg;

/// Perform register allocation on LIR instructions.
///
/// Returns the register mapping from VRegs to PhysRegs.
/// Pre-colors parameter registers: VReg(0)→PhysReg(0), etc.
#[must_use]
pub fn allocate_registers(
    instructions: &[LirInst],
    num_params: u32,
    max_regs: u32,
) -> RegMap {
    let ig = interference::InterferenceGraph::build_with_params(instructions, num_params);
    let coalesce_result = coalesce::coalesce(&ig, max_regs);
    let coloring_result = coloring::color(&ig, max_regs, num_params);

    let mut reg_map: RegMap = HashMap::new();

    for (vreg, phys) in &coloring_result.assignment {
        let final_vreg = find_representative(&coalesce_result.mapping, *vreg);
        reg_map.insert(final_vreg, *phys);
        reg_map.insert(*vreg, *phys);
    }

    for (coalesced, representative) in &coalesce_result.mapping {
        if let Some(&phys) = reg_map.get(representative) {
            reg_map.insert(*coalesced, phys);
        }
    }

    reg_map
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

/// Count the maximum physical register index used.
#[must_use]
pub fn max_register_used(reg_map: &RegMap) -> u32 {
    reg_map
        .values()
        .map(|p| u32::from(p.0) + 1)
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lir::operand::PhysReg;

    #[test]
    fn test_allocate_registers_simple() {
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

        let reg_map = allocate_registers(&insts, 0, 32);
        assert!(reg_map.contains_key(&VReg(0)));
        assert!(reg_map.contains_key(&VReg(1)));
        assert!(reg_map.contains_key(&VReg(2)));
    }

    #[test]
    fn test_allocate_with_params() {
        let insts = vec![
            LirInst::Iadd {
                dest: VReg(2),
                src1: VReg(0),
                src2: VReg(1),
            },
            LirInst::Halt,
        ];

        let reg_map = allocate_registers(&insts, 2, 32);
        assert_eq!(reg_map[&VReg(0)], PhysReg(0));
        assert_eq!(reg_map[&VReg(1)], PhysReg(1));
    }

    #[test]
    fn test_max_register_used() {
        let mut map = RegMap::new();
        map.insert(VReg(0), PhysReg(5));
        map.insert(VReg(1), PhysReg(10));
        assert_eq!(max_register_used(&map), 11);
    }
}
