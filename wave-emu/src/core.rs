// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Core simulation engine. Manages waves within a workgroup, coordinates barrier
//!
//! synchronization, schedules wave execution, and drives the instruction executor.
//! A single Core instance handles one workgroup's complete execution.

use crate::barrier::BarrierManager;
use crate::executor::{Executor, StepResult};
use crate::memory::{DeviceMemory, LocalMemory};
use crate::scheduler::Scheduler;
use crate::stats::ExecutionStats;
use crate::wave::Wave;
use crate::EmulatorConfig;
use crate::EmulatorError;

pub struct Core<'a> {
    waves: Vec<Wave>,
    local_memory: LocalMemory,
    device_memory: &'a mut DeviceMemory,
    scheduler: Scheduler,
    barrier_manager: BarrierManager,
    executor: Executor<'a>,
    stats: ExecutionStats,
    workgroup_id: [u32; 3],
    max_instructions: u64,
    instructions_executed: u64,
}

impl<'a> Core<'a> {
    pub fn new(
        config: &EmulatorConfig,
        code: &'a [u8],
        device_memory: &'a mut DeviceMemory,
        workgroup_id: [u32; 3],
    ) -> Self {
        let total_threads =
            config.workgroup_dim[0] * config.workgroup_dim[1] * config.workgroup_dim[2];
        let num_waves = total_threads.div_ceil(config.wave_width);

        let mut waves = Vec::with_capacity(num_waves as usize);
        for wave_id in 0..num_waves {
            let base_thread_index = wave_id * config.wave_width;
            let wave = Wave::new(
                config.wave_width,
                config.register_count,
                wave_id,
                workgroup_id,
                config.workgroup_dim,
                config.grid_dim,
                base_thread_index,
                total_threads,
                num_waves,
            );
            waves.push(wave);
        }

        // Apply initial register values to all threads in all waves
        for wave in &mut waves {
            for thread in &mut wave.threads {
                for &(reg, val) in &config.initial_registers {
                    thread.write_register(reg, val);
                }
            }
        }

        let local_memory = LocalMemory::new(config.local_memory_size);
        let scheduler = Scheduler::new(num_waves as usize);
        let barrier_manager = BarrierManager::new(num_waves);
        let executor = Executor::new(code, config.trace_enabled, workgroup_id);

        Self {
            waves,
            local_memory,
            device_memory,
            scheduler,
            barrier_manager,
            executor,
            stats: ExecutionStats::new(),
            workgroup_id,
            max_instructions: config.max_instructions,
            instructions_executed: 0,
        }
    }

    pub fn run(&mut self) -> Result<ExecutionStats, EmulatorError> {
        for _wave in &self.waves {
            self.stats.record_wave();
        }

        loop {
            if self.scheduler.all_halted(&self.waves) {
                break;
            }

            if self.barrier_manager.all_at_barrier() {
                self.barrier_manager.check_and_release(&mut self.waves);
                self.stats.record_barrier();
                continue;
            }

            let wave_index = match self.scheduler.pick_next_ready(&self.waves) {
                Some(idx) => idx,
                None => {
                    if self.scheduler.is_deadlocked(&self.waves) {
                        return Err(EmulatorError::Deadlock {
                            message: format!(
                                "workgroup ({},{},{}) deadlocked - waves waiting at different barriers",
                                self.workgroup_id[0], self.workgroup_id[1], self.workgroup_id[2]
                            ),
                        });
                    }
                    break;
                }
            };

            let wave = &mut self.waves[wave_index];
            let pc_before_step = wave.pc;

            let result = self.executor.step(
                wave,
                &mut self.local_memory,
                self.device_memory,
                &mut self.stats,
            )?;

            self.instructions_executed += 1;

            if self.max_instructions > 0 && self.instructions_executed > self.max_instructions {
                return Err(EmulatorError::InstructionLimitExceeded {
                    limit: self.max_instructions,
                    executed: self.instructions_executed,
                    pc: pc_before_step,
                });
            }

            match result {
                StepResult::Continue => {}
                StepResult::Halted => {}
                StepResult::Barrier => {
                    self.barrier_manager.handle_barrier(wave);
                }
            }
        }

        Ok(self.stats.clone())
    }

    pub fn waves(&self) -> &[Wave] {
        &self.waves
    }

    pub fn local_memory(&self) -> &LocalMemory {
        &self.local_memory
    }

    pub fn stats(&self) -> &ExecutionStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::{SyncOp, SYNC_MODIFIER_OFFSET};

    fn encode_halt() -> Vec<u8> {
        let halt_mod = SyncOp::Halt as u32 + u32::from(SYNC_MODIFIER_OFFSET);
        let word = (0x3Fu32 << 24) | (halt_mod << 4);
        word.to_le_bytes().to_vec()
    }

    fn encode_mov_imm(rd: u8, imm: u32) -> Vec<u8> {
        let word0 = (0x41u32 << 24) | ((u32::from(rd) & 0xFF) << 16) | (1u32 << 4);
        let mut code = word0.to_le_bytes().to_vec();
        code.extend_from_slice(&imm.to_le_bytes());
        code
    }

    #[test]
    fn test_core_single_wave_halt() {
        let code = encode_halt();

        let config = EmulatorConfig {
            grid_dim: [1, 1, 1],
            workgroup_dim: [32, 1, 1],
            register_count: 32,
            local_memory_size: 1024,
            device_memory_size: 1024,
            wave_width: 32,
            ..Default::default()
        };

        let mut device_memory = DeviceMemory::new(1024);
        let mut core = Core::new(&config, &code, &mut device_memory, [0, 0, 0]);

        let stats = core.run().unwrap();
        assert_eq!(stats.waves_executed, 1);
        assert!(core.waves[0].is_halted());
    }

    #[test]
    fn test_core_two_waves() {
        let code = encode_halt();

        let config = EmulatorConfig {
            grid_dim: [1, 1, 1],
            workgroup_dim: [64, 1, 1],
            register_count: 32,
            local_memory_size: 1024,
            device_memory_size: 1024,
            wave_width: 32,
            ..Default::default()
        };

        let mut device_memory = DeviceMemory::new(1024);
        let mut core = Core::new(&config, &code, &mut device_memory, [0, 0, 0]);

        let stats = core.run().unwrap();
        assert_eq!(stats.waves_executed, 2);
        assert!(core.waves[0].is_halted());
        assert!(core.waves[1].is_halted());
    }

    #[test]
    fn test_core_mov_imm_then_halt() {
        let mut code = encode_mov_imm(5, 0x12345678);
        code.extend_from_slice(&encode_halt());

        let config = EmulatorConfig {
            grid_dim: [1, 1, 1],
            workgroup_dim: [4, 1, 1],
            register_count: 32,
            local_memory_size: 1024,
            device_memory_size: 1024,
            wave_width: 4,
            ..Default::default()
        };

        let mut device_memory = DeviceMemory::new(1024);
        let mut core = Core::new(&config, &code, &mut device_memory, [0, 0, 0]);

        core.run().unwrap();

        for thread in &core.waves[0].threads {
            assert_eq!(thread.read_register(5), 0x12345678);
        }
    }

    #[test]
    fn test_core_thread_ids() {
        let code = encode_halt();

        let config = EmulatorConfig {
            grid_dim: [2, 2, 1],
            workgroup_dim: [8, 4, 2],
            register_count: 32,
            local_memory_size: 1024,
            device_memory_size: 1024,
            wave_width: 32,
            ..Default::default()
        };

        let mut device_memory = DeviceMemory::new(1024);
        let core = Core::new(&config, &code, &mut device_memory, [1, 0, 0]);

        assert_eq!(
            core.waves[0].threads[0].special_registers.workgroup_id,
            [1, 0, 0]
        );
        assert_eq!(
            core.waves[0].threads[0].special_registers.thread_id,
            [0, 0, 0]
        );
        assert_eq!(
            core.waves[0].threads[8].special_registers.thread_id,
            [0, 1, 0]
        );
    }

    fn encode_mov_sr(rd: u8, sr_index: u8) -> Vec<u8> {
        let word = (0x41u32 << 24)
            | ((u32::from(rd) & 0xFF) << 16)
            | ((u32::from(sr_index) & 0xFF) << 8)
            | (2u32 << 4);
        word.to_le_bytes().to_vec()
    }

    #[test]
    fn test_core_mov_sr_lane_id() {
        let mut code = encode_mov_sr(0, 4);
        code.extend_from_slice(&encode_halt());

        let config = EmulatorConfig {
            grid_dim: [1, 1, 1],
            workgroup_dim: [4, 1, 1],
            register_count: 32,
            local_memory_size: 1024,
            device_memory_size: 1024,
            wave_width: 4,
            ..Default::default()
        };

        let mut device_memory = DeviceMemory::new(1024);
        let mut core = Core::new(&config, &code, &mut device_memory, [0, 0, 0]);

        core.run().unwrap();

        for (i, thread) in core.waves[0].threads.iter().enumerate() {
            assert_eq!(
                thread.read_register(0),
                i as u32,
                "Thread {} should have lane_id {} in r0",
                i,
                i
            );
        }
    }

    fn encode_device_store_u32(addr_reg: u8, value_reg: u8) -> Vec<u8> {
        let word0 = ((0x39u32) << 24)
            | ((u32::from(addr_reg) & 0xFF) << 8)
            | ((2u32) << 4);
        let word1 = (u32::from(value_reg) & 0xFF) << 24;
        let mut code = word0.to_le_bytes().to_vec();
        code.extend_from_slice(&word1.to_le_bytes());
        code
    }

    fn encode_shl(rd: u8, rs1: u8, rs2: u8) -> Vec<u8> {
        let word0 = ((0x24u32) << 24)
            | ((u32::from(rd) & 0xFF) << 16)
            | ((u32::from(rs1) & 0xFF) << 8);
        let word1 = (u32::from(rs2) & 0xFF) << 24;
        let mut code = word0.to_le_bytes().to_vec();
        code.extend_from_slice(&word1.to_le_bytes());
        code
    }

    #[test]
    fn test_core_mov_sr_and_device_store() {
        let mut code = encode_mov_sr(0, 4); // r0 = lane_id
        code.extend_from_slice(&encode_mov_imm(1, 2)); // r1 = 2
        code.extend_from_slice(&encode_shl(2, 0, 1)); // r2 = r0 << r1 = lane_id * 4
        code.extend_from_slice(&encode_device_store_u32(2, 0)); // store r0 at addr r2
        code.extend_from_slice(&encode_halt());

        let config = EmulatorConfig {
            grid_dim: [1, 1, 1],
            workgroup_dim: [4, 1, 1],
            register_count: 32,
            local_memory_size: 1024,
            device_memory_size: 1024,
            wave_width: 4,
            ..Default::default()
        };

        let mut device_memory = DeviceMemory::new(1024);
        let mut core = Core::new(&config, &code, &mut device_memory, [0, 0, 0]);

        core.run().unwrap();

        for i in 0..4 {
            let addr = i * 4;
            let value = device_memory.read_u32(addr).unwrap();
            assert_eq!(value, i as u32, "Address {} should contain {}", addr, i);
        }
    }
}
