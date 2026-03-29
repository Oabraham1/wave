// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Spill code generation for register allocation.
//!
//! When a virtual register cannot be assigned a physical register,
//! spill code is inserted: stores before definitions and loads before uses.
//! The spilled value lives in local (scratchpad) memory.

use crate::lir::instruction::LirInst;
use crate::lir::operand::{MemWidth, VReg};

/// Generate spill code for a set of spilled virtual registers.
///
/// Inserts local stores after definitions and local loads before uses
/// of spilled registers, using a fresh `VReg` for each reload.
pub fn insert_spill_code(
    instructions: &mut Vec<LirInst>,
    spilled: &[VReg],
    next_vreg: &mut u32,
    spill_slot_base: u32,
) -> u32 {
    if spilled.is_empty() {
        return 0;
    }

    let mut spill_offsets: std::collections::HashMap<VReg, u32> = std::collections::HashMap::new();
    for (i, &vreg) in spilled.iter().enumerate() {
        spill_offsets.insert(vreg, spill_slot_base + (i as u32) * 4);
    }

    let mut new_insts: Vec<LirInst> = Vec::new();
    let mut spill_count = 0u32;

    for inst in instructions.iter() {
        let mut reload_map: std::collections::HashMap<VReg, VReg> =
            std::collections::HashMap::new();

        for src in inst.src_vregs() {
            if let Some(&offset) = spill_offsets.get(&src) {
                let reload_vreg = VReg(*next_vreg);
                *next_vreg += 1;

                let addr_vreg = VReg(*next_vreg);
                *next_vreg += 1;

                new_insts.push(LirInst::MovImm {
                    dest: addr_vreg,
                    value: offset,
                });
                new_insts.push(LirInst::LocalLoad {
                    dest: reload_vreg,
                    addr: addr_vreg,
                    width: MemWidth::W32,
                });
                reload_map.insert(src, reload_vreg);
                spill_count += 1;
            }
        }

        new_insts.push(inst.clone());

        if let Some(dest) = inst.dest_vreg() {
            if let Some(&offset) = spill_offsets.get(&dest) {
                let addr_vreg = VReg(*next_vreg);
                *next_vreg += 1;

                new_insts.push(LirInst::MovImm {
                    dest: addr_vreg,
                    value: offset,
                });
                new_insts.push(LirInst::LocalStore {
                    addr: addr_vreg,
                    value: dest,
                    width: MemWidth::W32,
                });
                spill_count += 1;
            }
        }
    }

    *instructions = new_insts;
    spill_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_spills_no_change() {
        let mut insts = vec![
            LirInst::MovImm {
                dest: VReg(0),
                value: 42,
            },
            LirInst::Halt,
        ];
        let original_len = insts.len();
        let mut next_vreg = 1;
        let count = insert_spill_code(&mut insts, &[], &mut next_vreg, 0);
        assert_eq!(count, 0);
        assert_eq!(insts.len(), original_len);
    }

    #[test]
    fn test_spill_inserts_stores() {
        let mut insts = vec![
            LirInst::MovImm {
                dest: VReg(0),
                value: 42,
            },
            LirInst::Halt,
        ];
        let mut next_vreg = 1;
        let count = insert_spill_code(&mut insts, &[VReg(0)], &mut next_vreg, 0);
        assert!(count > 0);
        assert!(insts.len() > 2);
        let has_store = insts
            .iter()
            .any(|i| matches!(i, LirInst::LocalStore { .. }));
        assert!(has_store);
    }
}
