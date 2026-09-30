// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Track wave arrivals at a workgroup barrier and release them when the workgroup is
//! synchronized.

use crate::wave::{Wave, WaveStatus};

#[derive(Debug)]
pub struct BarrierState {
    waves_at_barrier: u64,
    total_waves: u32,
}

impl BarrierState {
    #[must_use]
    pub fn new(total_waves: u32) -> Self {
        Self {
            waves_at_barrier: 0,
            total_waves,
        }
    }

    pub fn wave_reached_barrier(&mut self, wave_id: u32) {
        if wave_id < 64 {
            self.waves_at_barrier |= 1u64 << wave_id;
        }
    }

    #[must_use]
    pub fn all_waves_at_barrier(&self) -> bool {
        let expected_mask = if self.total_waves >= 64 {
            u64::MAX
        } else {
            (1u64 << self.total_waves) - 1
        };
        self.waves_at_barrier == expected_mask
    }

    pub fn reset(&mut self) {
        self.waves_at_barrier = 0;
    }

    #[must_use]
    pub fn waiting_count(&self) -> u32 {
        self.waves_at_barrier.count_ones()
    }
}

#[derive(Debug)]
pub struct BarrierManager {
    state: BarrierState,
    barrier_pc: Option<u32>,
}

impl BarrierManager {
    #[must_use]
    pub fn new(total_waves: u32) -> Self {
        Self {
            state: BarrierState::new(total_waves),
            barrier_pc: None,
        }
    }

    pub fn handle_barrier(&mut self, wave: &mut Wave) {
        wave.suspend();
        self.state.wave_reached_barrier(wave.wave_id);

        if self.barrier_pc.is_none() {
            self.barrier_pc = Some(wave.pc);
        }
    }

    pub fn check_and_release(&mut self, waves: &mut [Wave]) -> bool {
        if !self.state.all_waves_at_barrier() {
            return false;
        }

        for wave in waves.iter_mut() {
            if wave.status == WaveStatus::Suspended {
                wave.resume();
                wave.advance_pc(4);
            }
        }

        self.state.reset();
        self.barrier_pc = None;
        true
    }

    #[must_use]
    pub fn is_wave_at_barrier(&self, wave_id: u32) -> bool {
        if wave_id < 64 {
            (self.state.waves_at_barrier & (1u64 << wave_id)) != 0
        } else {
            false
        }
    }

    #[must_use]
    pub fn waiting_count(&self) -> u32 {
        self.state.waiting_count()
    }

    #[must_use]
    pub fn all_at_barrier(&self) -> bool {
        self.state.all_waves_at_barrier()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_wave(wave_id: u32) -> Wave {
        Wave::new(crate::wave::WaveConfig {
            wave_width: 32,
            register_count: 32,
            wave_id,
            workgroup_id: [0, 0, 0],
            workgroup_size: [64, 1, 1],
            grid_size: [1, 1, 1],
            base_thread_index: wave_id * 32,
            total_threads_in_workgroup: 64,
            num_waves: 2,
        })
    }

    #[test]
    fn test_barrier_single_wave() {
        let mut manager = BarrierManager::new(1);
        let mut wave = create_test_wave(0);
        wave.pc = 0x100;

        manager.handle_barrier(&mut wave);
        assert_eq!(wave.status, WaveStatus::Suspended);
        assert!(manager.is_wave_at_barrier(0));

        let mut workgroup = vec![wave];
        let released = manager.check_and_release(&mut workgroup);
        assert!(released);
        assert_eq!(workgroup[0].status, WaveStatus::Ready);
        assert_eq!(workgroup[0].pc, 0x104);
    }

    #[test]
    fn test_barrier_two_waves() {
        let mut manager = BarrierManager::new(2);
        let mut wave0 = create_test_wave(0);
        let mut wave1 = create_test_wave(1);
        wave0.pc = 0x100;
        wave1.pc = 0x100;

        manager.handle_barrier(&mut wave0);
        assert!(!manager.all_at_barrier());
        assert_eq!(manager.waiting_count(), 1);

        manager.handle_barrier(&mut wave1);
        assert!(manager.all_at_barrier());
        assert_eq!(manager.waiting_count(), 2);

        let mut workgroup = vec![wave0, wave1];
        let released = manager.check_and_release(&mut workgroup);
        assert!(released);

        for wave in &workgroup {
            assert_eq!(wave.status, WaveStatus::Ready);
            assert_eq!(wave.pc, 0x104);
        }
    }

    #[test]
    fn test_barrier_partial_arrival() {
        let mut manager = BarrierManager::new(4);
        let mut wave0 = create_test_wave(0);
        let mut wave1 = create_test_wave(1);
        wave0.pc = 0x100;
        wave1.pc = 0x100;

        manager.handle_barrier(&mut wave0);
        manager.handle_barrier(&mut wave1);

        assert!(!manager.all_at_barrier());
        assert_eq!(manager.waiting_count(), 2);

        let mut workgroup = vec![wave0, wave1];
        let released = manager.check_and_release(&mut workgroup);
        assert!(!released);
    }

    #[test]
    fn test_barrier_reset_after_release() {
        let mut manager = BarrierManager::new(1);
        let mut wave = create_test_wave(0);
        wave.pc = 0x100;

        manager.handle_barrier(&mut wave);

        let mut workgroup = vec![wave];
        manager.check_and_release(&mut workgroup);

        assert_eq!(manager.waiting_count(), 0);
        assert!(!manager.is_wave_at_barrier(0));
    }
}
