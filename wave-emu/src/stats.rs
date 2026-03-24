// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Execution statistics tracking. Counts instructions by category (integer, float,
//!
//! memory, control, wave ops, atomics), memory accesses, barriers, and divergent
//! branches. Aggregated across waves and workgroups for final reporting.

#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    pub instructions_executed: u64,
    pub integer_ops: u64,
    pub float_ops: u64,
    pub memory_ops: u64,
    pub control_ops: u64,
    pub wave_ops: u64,
    pub atomic_ops: u64,

    pub device_loads: u64,
    pub device_load_bytes: u64,
    pub device_stores: u64,
    pub device_store_bytes: u64,

    pub local_loads: u64,
    pub local_load_bytes: u64,
    pub local_stores: u64,
    pub local_store_bytes: u64,

    pub barriers: u64,
    pub divergent_branches: u64,

    pub workgroups_executed: u64,
    pub waves_executed: u64,
}

impl ExecutionStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_instruction(&mut self, category: InstructionCategory) {
        self.instructions_executed += 1;
        match category {
            InstructionCategory::Integer => self.integer_ops += 1,
            InstructionCategory::Float => self.float_ops += 1,
            InstructionCategory::Memory => self.memory_ops += 1,
            InstructionCategory::Control => self.control_ops += 1,
            InstructionCategory::WaveOp => self.wave_ops += 1,
            InstructionCategory::Atomic => self.atomic_ops += 1,
        }
    }

    pub fn record_device_load(&mut self, bytes: u64) {
        self.device_loads += 1;
        self.device_load_bytes += bytes;
    }

    pub fn record_device_store(&mut self, bytes: u64) {
        self.device_stores += 1;
        self.device_store_bytes += bytes;
    }

    pub fn record_local_load(&mut self, bytes: u64) {
        self.local_loads += 1;
        self.local_load_bytes += bytes;
    }

    pub fn record_local_store(&mut self, bytes: u64) {
        self.local_stores += 1;
        self.local_store_bytes += bytes;
    }

    pub fn record_barrier(&mut self) {
        self.barriers += 1;
    }

    pub fn record_divergent_branch(&mut self) {
        self.divergent_branches += 1;
    }

    pub fn record_wave(&mut self) {
        self.waves_executed += 1;
    }

    pub fn merge(&mut self, other: &ExecutionStats) {
        self.instructions_executed += other.instructions_executed;
        self.integer_ops += other.integer_ops;
        self.float_ops += other.float_ops;
        self.memory_ops += other.memory_ops;
        self.control_ops += other.control_ops;
        self.wave_ops += other.wave_ops;
        self.atomic_ops += other.atomic_ops;

        self.device_loads += other.device_loads;
        self.device_load_bytes += other.device_load_bytes;
        self.device_stores += other.device_stores;
        self.device_store_bytes += other.device_store_bytes;

        self.local_loads += other.local_loads;
        self.local_load_bytes += other.local_load_bytes;
        self.local_stores += other.local_stores;
        self.local_store_bytes += other.local_store_bytes;

        self.barriers += other.barriers;
        self.divergent_branches += other.divergent_branches;

        self.waves_executed += other.waves_executed;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstructionCategory {
    Integer,
    Float,
    Memory,
    Control,
    WaveOp,
    Atomic,
}

pub struct TraceWriter {
    enabled: bool,
}

impl TraceWriter {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub fn trace_instruction(
        &self,
        workgroup_id: [u32; 3],
        wave_id: u32,
        pc: u32,
        disasm: &str,
    ) {
        if self.enabled {
            eprintln!(
                "wg({},{},{}) wave[{}] pc=0x{:04x}: {}",
                workgroup_id[0], workgroup_id[1], workgroup_id[2],
                wave_id,
                pc,
                disasm
            );
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl Default for TraceWriter {
    fn default() -> Self {
        Self::new(false)
    }
}

impl std::fmt::Display for ExecutionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Execution Statistics:")?;
        writeln!(f, "  Instructions executed: {}", self.instructions_executed)?;
        writeln!(f, "    Integer ops:         {}", self.integer_ops)?;
        writeln!(f, "    Float ops:           {}", self.float_ops)?;
        writeln!(f, "    Memory ops:          {}", self.memory_ops)?;
        writeln!(f, "    Control ops:         {}", self.control_ops)?;
        writeln!(f, "    Wave ops:            {}", self.wave_ops)?;
        writeln!(f, "    Atomic ops:          {}", self.atomic_ops)?;
        writeln!(f)?;
        writeln!(f, "  Device memory:")?;
        writeln!(f, "    Loads:  {} ({} bytes)", self.device_loads, self.device_load_bytes)?;
        writeln!(f, "    Stores: {} ({} bytes)", self.device_stores, self.device_store_bytes)?;
        writeln!(f)?;
        writeln!(f, "  Local memory:")?;
        writeln!(f, "    Loads:  {} ({} bytes)", self.local_loads, self.local_load_bytes)?;
        writeln!(f, "    Stores: {} ({} bytes)", self.local_stores, self.local_store_bytes)?;
        writeln!(f)?;
        writeln!(f, "  Barriers: {}", self.barriers)?;
        writeln!(f, "  Divergent branches: {}", self.divergent_branches)?;
        writeln!(f)?;
        writeln!(f, "  Workgroups executed: {}", self.workgroups_executed)?;
        writeln!(f, "  Waves executed: {}", self.waves_executed)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_record_instruction() {
        let mut stats = ExecutionStats::new();
        stats.record_instruction(InstructionCategory::Integer);
        stats.record_instruction(InstructionCategory::Float);
        stats.record_instruction(InstructionCategory::Float);

        assert_eq!(stats.instructions_executed, 3);
        assert_eq!(stats.integer_ops, 1);
        assert_eq!(stats.float_ops, 2);
    }

    #[test]
    fn test_stats_merge() {
        let mut stats1 = ExecutionStats::new();
        stats1.instructions_executed = 100;
        stats1.device_loads = 50;

        let mut stats2 = ExecutionStats::new();
        stats2.instructions_executed = 200;
        stats2.device_loads = 30;

        stats1.merge(&stats2);

        assert_eq!(stats1.instructions_executed, 300);
        assert_eq!(stats1.device_loads, 80);
    }

    #[test]
    fn test_stats_display() {
        let stats = ExecutionStats::new();
        let output = format!("{stats}");
        assert!(output.contains("Execution Statistics:"));
    }
}
