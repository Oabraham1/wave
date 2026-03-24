// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Wave scheduler for multi-wave workgroups. Round-robin selection of ready waves,
//!
//! skipping suspended (at barrier) and halted waves. Detects deadlock when no
//! wave is ready but not all waves are halted.

use crate::wave::{Wave, WaveStatus};

pub struct Scheduler {
    last_wave_index: usize,
    wave_count: usize,
}

impl Scheduler {
    pub fn new(wave_count: usize) -> Self {
        Self {
            last_wave_index: 0,
            wave_count,
        }
    }

    pub fn pick_next_ready(&mut self, waves: &[Wave]) -> Option<usize> {
        if waves.is_empty() {
            return None;
        }

        for offset in 0..self.wave_count {
            let index = (self.last_wave_index + offset) % self.wave_count;
            if index < waves.len() && waves[index].is_ready() {
                self.last_wave_index = (index + 1) % self.wave_count;
                return Some(index);
            }
        }

        None
    }

    pub fn all_halted(&self, waves: &[Wave]) -> bool {
        waves.iter().all(|w| w.is_halted())
    }

    pub fn any_ready(&self, waves: &[Wave]) -> bool {
        waves.iter().any(|w| w.is_ready())
    }

    pub fn count_by_status(&self, waves: &[Wave]) -> (usize, usize, usize) {
        let ready = waves.iter().filter(|w| w.status == WaveStatus::Ready).count();
        let suspended = waves.iter().filter(|w| w.status == WaveStatus::Suspended).count();
        let halted = waves.iter().filter(|w| w.status == WaveStatus::Halted).count();
        (ready, suspended, halted)
    }

    pub fn is_deadlocked(&self, waves: &[Wave]) -> bool {
        let (ready, suspended, _halted) = self.count_by_status(waves);
        ready == 0 && suspended > 0
    }

    pub fn reset(&mut self) {
        self.last_wave_index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_waves(count: usize) -> Vec<Wave> {
        (0..count)
            .map(|i| {
                Wave::new(
                    32,
                    32,
                    i as u32,
                    [0, 0, 0],
                    [(count * 32) as u32, 1, 1],
                    [1, 1, 1],
                    (i * 32) as u32,
                    (count * 32) as u32,
                    count as u32,
                )
            })
            .collect()
    }

    #[test]
    fn test_scheduler_round_robin() {
        let waves = create_test_waves(4);
        let mut scheduler = Scheduler::new(4);

        let first = scheduler.pick_next_ready(&waves);
        assert_eq!(first, Some(0));

        let second = scheduler.pick_next_ready(&waves);
        assert_eq!(second, Some(1));

        let third = scheduler.pick_next_ready(&waves);
        assert_eq!(third, Some(2));

        let fourth = scheduler.pick_next_ready(&waves);
        assert_eq!(fourth, Some(3));

        let fifth = scheduler.pick_next_ready(&waves);
        assert_eq!(fifth, Some(0));
    }

    #[test]
    fn test_scheduler_skip_halted() {
        let mut waves = create_test_waves(4);
        waves[1].halt();
        waves[2].halt();

        let mut scheduler = Scheduler::new(4);

        let first = scheduler.pick_next_ready(&waves);
        assert_eq!(first, Some(0));

        let second = scheduler.pick_next_ready(&waves);
        assert_eq!(second, Some(3));

        let third = scheduler.pick_next_ready(&waves);
        assert_eq!(third, Some(0));
    }

    #[test]
    fn test_scheduler_skip_suspended() {
        let mut waves = create_test_waves(3);
        waves[1].suspend();

        let mut scheduler = Scheduler::new(3);

        let first = scheduler.pick_next_ready(&waves);
        assert_eq!(first, Some(0));

        let second = scheduler.pick_next_ready(&waves);
        assert_eq!(second, Some(2));

        let third = scheduler.pick_next_ready(&waves);
        assert_eq!(third, Some(0));
    }

    #[test]
    fn test_scheduler_all_halted() {
        let mut waves = create_test_waves(3);
        for wave in &mut waves {
            wave.halt();
        }

        let scheduler = Scheduler::new(3);
        assert!(scheduler.all_halted(&waves));
    }

    #[test]
    fn test_scheduler_none_ready() {
        let mut waves = create_test_waves(2);
        waves[0].halt();
        waves[1].suspend();

        let mut scheduler = Scheduler::new(2);
        assert_eq!(scheduler.pick_next_ready(&waves), None);
    }

    #[test]
    fn test_scheduler_deadlock_detection() {
        let mut waves = create_test_waves(3);
        waves[0].suspend();
        waves[1].suspend();
        waves[2].halt();

        let scheduler = Scheduler::new(3);
        assert!(scheduler.is_deadlocked(&waves));
    }

    #[test]
    fn test_scheduler_no_deadlock_when_all_halted() {
        let mut waves = create_test_waves(2);
        waves[0].halt();
        waves[1].halt();

        let scheduler = Scheduler::new(2);
        assert!(!scheduler.is_deadlocked(&waves));
    }

    #[test]
    fn test_scheduler_count_by_status() {
        let mut waves = create_test_waves(5);
        waves[0].halt();
        waves[1].suspend();
        waves[2].suspend();

        let scheduler = Scheduler::new(5);
        let (ready, suspended, halted) = scheduler.count_by_status(&waves);

        assert_eq!(ready, 2);
        assert_eq!(suspended, 2);
        assert_eq!(halted, 1);
    }
}
