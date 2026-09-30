// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Allocate physical registers by coloring the interference graph.

pub mod coalesce;
pub mod coloring;
pub mod interference;
pub mod live_range;
pub mod spill;

use crate::diagnostics::CompileError;
use crate::emit::wave_emit::RegMap;
use crate::lir::instruction::LirInst;

/// Perform register allocation on LIR instructions.
///
/// Returns the register mapping from `VReg`s to `PhysReg`s.
/// Parameter registers retain their argument indices.
///
/// # Examples
///
/// ```
/// use wave_compiler::lir::operand::{PhysReg, VReg};
/// use wave_compiler::regalloc::allocate_registers;
///
/// let mapping = allocate_registers(&[], 2, 32).expect("parameter registers fit");
/// assert_eq!(mapping[&VReg(1)], PhysReg(1));
/// ```
///
/// # Errors
///
/// Returns an error if the register limit is invalid or live values cannot fit.
pub fn allocate_registers(
    instructions: &[LirInst],
    num_params: u32,
    max_regs: u32,
) -> Result<RegMap, CompileError> {
    if max_regs == 0 || max_regs > 256 || num_params > max_regs {
        return Err(CompileError::RegAllocError {
            message: "register limit must be 1 to 256 and cover all kernel parameters".into(),
        });
    }
    let ig = interference::InterferenceGraph::build_with_params(instructions, num_params);
    let coloring_result = coloring::color(&ig, max_regs, num_params);
    if !coloring_result.spilled.is_empty() {
        return Err(CompileError::RegAllocError {
            message: format!("live values exceed the {max_regs}-register limit; automatic spilling is unavailable"),
        });
    }

    Ok(coloring_result.assignment)
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
    use crate::lir::operand::{MemWidth, PhysReg, VReg};

    #[test]
    fn test_allocate_registers_simple() {
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

        let reg_map = allocate_registers(&insts, 0, 32).unwrap();
        assert!(allocate_registers(&insts, 0, 1).is_err());
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

        let reg_map = allocate_registers(&insts, 2, 32).unwrap();
        assert_eq!(reg_map[&VReg(0)], PhysReg(0));
        assert_eq!(reg_map[&VReg(1)], PhysReg(1));
    }

    #[test]
    fn allocation_preserves_live_values_through_copies() -> Result<(), CompileError> {
        let instructions = [
            LirInst::MovReg {
                dest: VReg(2),
                src: VReg(1),
            },
            LirInst::MovImm {
                dest: VReg(3),
                value: 9,
            },
            LirInst::MovReg {
                dest: VReg(4),
                src: VReg(2),
            },
            LirInst::Iadd {
                dest: VReg(5),
                src1: VReg(3),
                src2: VReg(1),
            },
            LirInst::DeviceStore {
                addr: VReg(0),
                value: VReg(4),
                width: MemWidth::W32,
            },
            LirInst::DeviceStore {
                addr: VReg(0),
                value: VReg(5),
                width: MemWidth::W32,
            },
            LirInst::Halt,
        ];
        let mapping = allocate_registers(&instructions, 2, 6)?;
        assert_eq!(mapping[&VReg(0)], PhysReg(0));
        assert_eq!(mapping[&VReg(1)], PhysReg(1));

        let mut registers = [0_u32; 6];
        registers[0] = 32;
        registers[1] = 17;
        let mut writes = Vec::new();
        for instruction in &instructions {
            let index = |value| usize::from(mapping[&value].0);
            match *instruction {
                LirInst::MovImm { dest, value } => registers[index(dest)] = value,
                LirInst::MovReg { dest, src } => registers[index(dest)] = registers[index(src)],
                LirInst::Iadd { dest, src1, src2 } => {
                    registers[index(dest)] =
                        registers[index(src1)].wrapping_add(registers[index(src2)]);
                }
                LirInst::DeviceStore { addr, value, .. } => {
                    writes.push((registers[index(addr)], registers[index(value)]));
                }
                LirInst::Halt => break,
                _ => panic!("unsupported test instruction: {instruction:?}"),
            }
        }
        assert_eq!(writes, [(32, 17), (32, 26)]);
        assert!(matches!(
            allocate_registers(&instructions, 2, 2),
            Err(CompileError::RegAllocError { .. })
        ));
        Ok(())
    }

    #[test]
    fn loop_back_edges_preserve_parameter_values() -> Result<(), CompileError> {
        let instructions = [
            LirInst::Loop,
            LirInst::MovReg {
                dest: VReg(2),
                src: VReg(1),
            },
            LirInst::DeviceStore {
                addr: VReg(0),
                value: VReg(2),
                width: MemWidth::W32,
            },
            LirInst::MovImm {
                dest: VReg(3),
                value: 99,
            },
            LirInst::DeviceStore {
                addr: VReg(0),
                value: VReg(3),
                width: MemWidth::W32,
            },
            LirInst::Endloop,
        ];
        let mapping = allocate_registers(&instructions, 2, 3)?;
        let mut registers = [32_u32, 17, 0];
        let mut writes = Vec::new();
        for _ in 0..2 {
            for instruction in &instructions {
                let index = |value| usize::from(mapping[&value].0);
                match *instruction {
                    LirInst::MovReg { dest, src } => registers[index(dest)] = registers[index(src)],
                    LirInst::MovImm { dest, value } => registers[index(dest)] = value,
                    LirInst::DeviceStore { addr, value, .. } => {
                        writes.push((registers[index(addr)], registers[index(value)]));
                    }
                    LirInst::Loop | LirInst::Endloop => {}
                    _ => panic!("unsupported test instruction: {instruction:?}"),
                }
            }
        }
        assert_eq!(writes, [(32, 17), (32, 99), (32, 17), (32, 99)]);
        assert_eq!(registers[0..2], [32, 17]);
        assert_eq!(mapping[&VReg(2)], mapping[&VReg(3)]);
        Ok(())
    }

    #[test]
    fn test_max_register_used() {
        let mut map = RegMap::new();
        map.insert(VReg(0), PhysReg(5));
        map.insert(VReg(1), PhysReg(10));
        assert_eq!(max_register_used(&map), 11);
    }
}
