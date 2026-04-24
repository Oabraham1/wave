// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Per-thread execution state. Each thread has a register file (32 x u32), four
//!
//! predicate registers (p0-p3), and read-only special registers populated at
//! dispatch time (thread/wave/workgroup IDs, dimensions, etc).

use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct MmaFragments {
    pub a_fragments: HashMap<u8, Vec<f32>>,
    pub b_fragments: HashMap<u8, Vec<f32>>,
    pub c_fragments: HashMap<u8, Vec<f32>>,
}

#[derive(Debug)]
pub struct Thread {
    pub registers: Vec<u32>,
    pub predicates: [bool; 4],
    pub special_registers: SpecialRegisters,
    pub mma_fragments: MmaFragments,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SpecialRegisters {
    pub thread_id: [u32; 3],
    pub wave_id: u32,
    pub lane_id: u32,
    pub workgroup_id: [u32; 3],
    pub workgroup_size: [u32; 3],
    pub grid_size: [u32; 3],
    pub wave_width: u32,
    pub num_waves: u32,
    pub mma_supported: u32,
    pub mma_m: u32,
    pub mma_n: u32,
    pub mma_k: u32,
}

impl SpecialRegisters {
    pub fn get(&self, index: u8) -> u32 {
        match index {
            0 => self.thread_id[0],
            1 => self.thread_id[1],
            2 => self.thread_id[2],
            3 => self.wave_id,
            4 => self.lane_id,
            5 => self.workgroup_id[0],
            6 => self.workgroup_id[1],
            7 => self.workgroup_id[2],
            8 => self.workgroup_size[0],
            9 => self.workgroup_size[1],
            10 => self.workgroup_size[2],
            11 => self.grid_size[0],
            12 => self.grid_size[1],
            13 => self.grid_size[2],
            14 => self.wave_width,
            15 => self.num_waves,
            16 => self.mma_supported,
            17 => self.mma_m,
            18 => self.mma_n,
            19 => self.mma_k,
            _ => 0,
        }
    }
}

impl Thread {
    pub fn new(register_count: u32) -> Self {
        Self {
            registers: vec![0; register_count as usize],
            predicates: [false; 4],
            special_registers: SpecialRegisters::default(),
            mma_fragments: MmaFragments::default(),
        }
    }

    pub fn with_special_registers(register_count: u32, special: SpecialRegisters) -> Self {
        Self {
            registers: vec![0; register_count as usize],
            predicates: [false; 4],
            special_registers: special,
            mma_fragments: MmaFragments::default(),
        }
    }

    pub fn read_register(&self, index: u8) -> u32 {
        self.registers.get(index as usize).copied().unwrap_or(0)
    }

    pub fn write_register(&mut self, index: u8, value: u32) {
        if (index as usize) < self.registers.len() {
            self.registers[index as usize] = value;
        }
    }

    pub fn read_predicate(&self, index: u8) -> bool {
        self.predicates
            .get(index as usize)
            .copied()
            .unwrap_or(false)
    }

    pub fn write_predicate(&mut self, index: u8, value: bool) {
        if (index as usize) < self.predicates.len() {
            self.predicates[index as usize] = value;
        }
    }

    pub fn read_special(&self, index: u8) -> u32 {
        self.special_registers.get(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_new() {
        let thread = Thread::new(32);
        assert_eq!(thread.registers.len(), 32);
        assert_eq!(thread.predicates, [false; 4]);
    }

    #[test]
    fn test_thread_register_read_write() {
        let mut thread = Thread::new(32);
        thread.write_register(5, 0x12345678);
        assert_eq!(thread.read_register(5), 0x12345678);
    }

    #[test]
    fn test_thread_predicate_read_write() {
        let mut thread = Thread::new(32);
        thread.write_predicate(2, true);
        assert!(thread.read_predicate(2));
        assert!(!thread.read_predicate(0));
    }

    #[test]
    fn test_thread_special_registers() {
        let special = SpecialRegisters {
            thread_id: [10, 20, 30],
            wave_id: 2,
            lane_id: 15,
            workgroup_id: [1, 2, 3],
            workgroup_size: [64, 1, 1],
            grid_size: [4, 4, 1],
            wave_width: 32,
            num_waves: 2,
            mma_supported: 1,
            mma_m: 4,
            mma_n: 4,
            mma_k: 4,
        };
        let thread = Thread::with_special_registers(32, special);

        assert_eq!(thread.read_special(0), 10);
        assert_eq!(thread.read_special(1), 20);
        assert_eq!(thread.read_special(2), 30);
        assert_eq!(thread.read_special(3), 2);
        assert_eq!(thread.read_special(4), 15);
        assert_eq!(thread.read_special(5), 1);
        assert_eq!(thread.read_special(14), 32);
    }

    #[test]
    fn test_thread_out_of_bounds_register() {
        let thread = Thread::new(8);
        assert_eq!(thread.read_register(100), 0);
    }
}
