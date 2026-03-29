// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Wave state management. A wave contains W threads sharing a program counter.
//!
//! Active mask (u64 bitmask) tracks which threads execute each instruction.
//! Supports wave widths up to 64. Status tracks ready/suspended/halted state.

use crate::control_flow::ControlFlowManager;
use crate::thread::{SpecialRegisters, Thread};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveStatus {
    Ready,
    Suspended,
    Halted,
}

#[derive(Debug)]
pub struct Wave {
    pub threads: Vec<Thread>,
    pub pc: u32,
    pub active_mask: u64,
    pub status: WaveStatus,
    pub wave_width: u32,
    pub wave_id: u32,
    call_stack: Vec<u32>,
    pub control_flow: ControlFlowManager,
}

impl Wave {
    pub fn new(
        wave_width: u32,
        register_count: u32,
        wave_id: u32,
        workgroup_id: [u32; 3],
        workgroup_size: [u32; 3],
        grid_size: [u32; 3],
        base_thread_index: u32,
        total_threads_in_workgroup: u32,
        num_waves: u32,
    ) -> Self {
        let mut threads = Vec::with_capacity(wave_width as usize);

        for lane_id in 0..wave_width {
            let global_thread_index = base_thread_index + lane_id;

            let thread_id = Self::compute_thread_id(global_thread_index, workgroup_size);

            let special = SpecialRegisters {
                thread_id,
                wave_id,
                lane_id,
                workgroup_id,
                workgroup_size,
                grid_size,
                wave_width,
                num_waves,
            };

            threads.push(Thread::with_special_registers(register_count, special));
        }

        let active_threads =
            (total_threads_in_workgroup.saturating_sub(base_thread_index)).min(wave_width);
        let active_mask = if active_threads >= 64 {
            u64::MAX
        } else {
            (1u64 << active_threads) - 1
        };

        Self {
            threads,
            pc: 0,
            active_mask,
            status: WaveStatus::Ready,
            wave_width,
            wave_id,
            call_stack: Vec::with_capacity(8),
            control_flow: ControlFlowManager::new(),
        }
    }

    fn compute_thread_id(linear_index: u32, workgroup_size: [u32; 3]) -> [u32; 3] {
        let x = linear_index % workgroup_size[0];
        let y = (linear_index / workgroup_size[0]) % workgroup_size[1];
        let z = linear_index / (workgroup_size[0] * workgroup_size[1]);
        [x, y, z]
    }

    pub fn is_thread_active(&self, lane: u32) -> bool {
        if lane >= 64 {
            return false;
        }
        (self.active_mask & (1u64 << lane)) != 0
    }

    pub fn active_thread_count(&self) -> u32 {
        self.active_mask.count_ones()
    }

    pub fn set_thread_active(&mut self, lane: u32, active: bool) {
        if lane < 64 {
            if active {
                self.active_mask |= 1u64 << lane;
            } else {
                self.active_mask &= !(1u64 << lane);
            }
        }
    }

    pub fn push_call(&mut self, return_pc: u32) -> Result<(), &'static str> {
        if self.call_stack.len() >= 8 {
            return Err("call stack overflow");
        }
        self.call_stack.push(return_pc);
        Ok(())
    }

    pub fn pop_call(&mut self) -> Option<u32> {
        self.call_stack.pop()
    }

    pub fn call_depth(&self) -> usize {
        self.call_stack.len()
    }

    pub fn halt(&mut self) {
        self.status = WaveStatus::Halted;
        self.active_mask = 0;
    }

    pub fn suspend(&mut self) {
        self.status = WaveStatus::Suspended;
    }

    pub fn resume(&mut self) {
        if self.status == WaveStatus::Suspended {
            self.status = WaveStatus::Ready;
        }
    }

    pub fn is_halted(&self) -> bool {
        self.status == WaveStatus::Halted
    }

    pub fn is_ready(&self) -> bool {
        self.status == WaveStatus::Ready
    }

    pub fn advance_pc(&mut self, bytes: u32) {
        self.pc = self.pc.wrapping_add(bytes);
    }

    pub fn set_pc(&mut self, pc: u32) {
        self.pc = pc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wave_new() {
        let wave = Wave::new(32, 32, 0, [0, 0, 0], [64, 1, 1], [1, 1, 1], 0, 64, 2);
        assert_eq!(wave.threads.len(), 32);
        assert_eq!(wave.active_mask, 0xFFFF_FFFF);
        assert_eq!(wave.status, WaveStatus::Ready);
    }

    #[test]
    fn test_wave_partial_active() {
        let wave = Wave::new(32, 32, 1, [0, 0, 0], [48, 1, 1], [1, 1, 1], 32, 48, 2);
        assert_eq!(wave.active_mask, 0xFFFF);
        assert_eq!(wave.active_thread_count(), 16);
    }

    #[test]
    fn test_wave_thread_ids() {
        let wave = Wave::new(32, 32, 0, [1, 2, 3], [8, 4, 2], [4, 4, 1], 0, 64, 2);

        assert_eq!(wave.threads[0].special_registers.thread_id, [0, 0, 0]);
        assert_eq!(wave.threads[8].special_registers.thread_id, [0, 1, 0]);
        assert_eq!(wave.threads[16].special_registers.thread_id, [0, 2, 0]);
    }

    #[test]
    fn test_wave_lane_ids() {
        let wave = Wave::new(4, 32, 0, [0, 0, 0], [4, 1, 1], [1, 1, 1], 0, 4, 1);

        assert_eq!(wave.threads[0].special_registers.lane_id, 0);
        assert_eq!(wave.threads[1].special_registers.lane_id, 1);
        assert_eq!(wave.threads[2].special_registers.lane_id, 2);
        assert_eq!(wave.threads[3].special_registers.lane_id, 3);

        assert_eq!(wave.threads[0].read_special(4), 0);
        assert_eq!(wave.threads[1].read_special(4), 1);
        assert_eq!(wave.threads[2].read_special(4), 2);
        assert_eq!(wave.threads[3].read_special(4), 3);
    }

    #[test]
    fn test_wave_call_stack() {
        let mut wave = Wave::new(32, 32, 0, [0, 0, 0], [32, 1, 1], [1, 1, 1], 0, 32, 1);

        wave.push_call(0x100).unwrap();
        wave.push_call(0x200).unwrap();

        assert_eq!(wave.call_depth(), 2);
        assert_eq!(wave.pop_call(), Some(0x200));
        assert_eq!(wave.pop_call(), Some(0x100));
        assert_eq!(wave.pop_call(), None);
    }

    #[test]
    fn test_wave_halt() {
        let mut wave = Wave::new(32, 32, 0, [0, 0, 0], [32, 1, 1], [1, 1, 1], 0, 32, 1);
        wave.halt();

        assert!(wave.is_halted());
        assert_eq!(wave.active_mask, 0);
    }

    #[test]
    fn test_wave_suspend_resume() {
        let mut wave = Wave::new(32, 32, 0, [0, 0, 0], [32, 1, 1], [1, 1, 1], 0, 32, 1);

        wave.suspend();
        assert_eq!(wave.status, WaveStatus::Suspended);

        wave.resume();
        assert!(wave.is_ready());
    }

    #[test]
    fn test_wave_set_thread_active() {
        let mut wave = Wave::new(32, 32, 0, [0, 0, 0], [32, 1, 1], [1, 1, 1], 0, 32, 1);

        wave.set_thread_active(5, false);
        assert!(!wave.is_thread_active(5));
        assert!(wave.is_thread_active(4));

        wave.set_thread_active(5, true);
        assert!(wave.is_thread_active(5));
    }
}
