// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Memory subsystem implementation. DeviceMemory is global shared storage across
//!
//! all workgroups. LocalMemory is per-workgroup scratch space. Both support byte,
//! half, word, and double-word loads/stores with bounds checking.

use crate::EmulatorError;

#[derive(Debug)]
pub struct DeviceMemory {
    data: Vec<u8>,
}

impl DeviceMemory {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0; size],
        }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    fn check_bounds(&self, addr: u64, size: usize) -> Result<usize, EmulatorError> {
        let addr = addr as usize;
        if addr
            .checked_add(size)
            .is_none_or(|end| end > self.data.len())
        {
            return Err(EmulatorError::MemoryOutOfBounds {
                address: addr as u64,
            });
        }
        Ok(addr)
    }

    pub fn read_u8(&self, addr: u64) -> Result<u8, EmulatorError> {
        let addr = self.check_bounds(addr, 1)?;
        Ok(self.data[addr])
    }

    pub fn read_u16(&self, addr: u64) -> Result<u16, EmulatorError> {
        let addr = self.check_bounds(addr, 2)?;
        Ok(u16::from_le_bytes([self.data[addr], self.data[addr + 1]]))
    }

    pub fn read_u32(&self, addr: u64) -> Result<u32, EmulatorError> {
        let addr = self.check_bounds(addr, 4)?;
        Ok(u32::from_le_bytes([
            self.data[addr],
            self.data[addr + 1],
            self.data[addr + 2],
            self.data[addr + 3],
        ]))
    }

    pub fn read_u64(&self, addr: u64) -> Result<u64, EmulatorError> {
        let addr = self.check_bounds(addr, 8)?;
        Ok(u64::from_le_bytes([
            self.data[addr],
            self.data[addr + 1],
            self.data[addr + 2],
            self.data[addr + 3],
            self.data[addr + 4],
            self.data[addr + 5],
            self.data[addr + 6],
            self.data[addr + 7],
        ]))
    }

    pub fn read_u128(&self, addr: u64) -> Result<u128, EmulatorError> {
        let addr = self.check_bounds(addr, 16)?;
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&self.data[addr..addr + 16]);
        Ok(u128::from_le_bytes(bytes))
    }

    pub fn write_u8(&mut self, addr: u64, value: u8) -> Result<(), EmulatorError> {
        let addr = self.check_bounds(addr, 1)?;
        self.data[addr] = value;
        Ok(())
    }

    pub fn write_u16(&mut self, addr: u64, value: u16) -> Result<(), EmulatorError> {
        let addr = self.check_bounds(addr, 2)?;
        let bytes = value.to_le_bytes();
        self.data[addr..addr + 2].copy_from_slice(&bytes);
        Ok(())
    }

    pub fn write_u32(&mut self, addr: u64, value: u32) -> Result<(), EmulatorError> {
        let addr = self.check_bounds(addr, 4)?;
        let bytes = value.to_le_bytes();
        self.data[addr..addr + 4].copy_from_slice(&bytes);
        Ok(())
    }

    pub fn write_u64(&mut self, addr: u64, value: u64) -> Result<(), EmulatorError> {
        let addr = self.check_bounds(addr, 8)?;
        let bytes = value.to_le_bytes();
        self.data[addr..addr + 8].copy_from_slice(&bytes);
        Ok(())
    }

    pub fn write_u128(&mut self, addr: u64, value: u128) -> Result<(), EmulatorError> {
        let addr = self.check_bounds(addr, 16)?;
        let bytes = value.to_le_bytes();
        self.data[addr..addr + 16].copy_from_slice(&bytes);
        Ok(())
    }

    pub fn write_slice(&mut self, offset: u64, data: &[u8]) -> Result<(), EmulatorError> {
        let addr = self.check_bounds(offset, data.len())?;
        self.data[addr..addr + data.len()].copy_from_slice(data);
        Ok(())
    }

    pub fn atomic_add(&mut self, addr: u64, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old.wrapping_add(value))?;
        Ok(old)
    }

    pub fn atomic_sub(&mut self, addr: u64, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old.wrapping_sub(value))?;
        Ok(old)
    }

    pub fn atomic_min(&mut self, addr: u64, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old.min(value))?;
        Ok(old)
    }

    pub fn atomic_max(&mut self, addr: u64, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old.max(value))?;
        Ok(old)
    }

    pub fn atomic_and(&mut self, addr: u64, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old & value)?;
        Ok(old)
    }

    pub fn atomic_or(&mut self, addr: u64, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old | value)?;
        Ok(old)
    }

    pub fn atomic_xor(&mut self, addr: u64, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old ^ value)?;
        Ok(old)
    }

    pub fn atomic_exchange(&mut self, addr: u64, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, value)?;
        Ok(old)
    }

    pub fn atomic_cas(
        &mut self,
        addr: u64,
        expected: u32,
        desired: u32,
    ) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        if old == expected {
            self.write_u32(addr, desired)?;
        }
        Ok(old)
    }

    pub fn atomic_add_i32(&mut self, addr: u64, value: i32) -> Result<i32, EmulatorError> {
        let old = self.read_u32(addr)? as i32;
        self.write_u32(addr, old.wrapping_add(value) as u32)?;
        Ok(old)
    }

    pub fn atomic_min_i32(&mut self, addr: u64, value: i32) -> Result<i32, EmulatorError> {
        let old = self.read_u32(addr)? as i32;
        self.write_u32(addr, old.min(value) as u32)?;
        Ok(old)
    }

    pub fn atomic_max_i32(&mut self, addr: u64, value: i32) -> Result<i32, EmulatorError> {
        let old = self.read_u32(addr)? as i32;
        self.write_u32(addr, old.max(value) as u32)?;
        Ok(old)
    }

    pub fn atomic_add_f32(&mut self, addr: u64, value: f32) -> Result<f32, EmulatorError> {
        let old_bits = self.read_u32(addr)?;
        let old = f32::from_bits(old_bits);
        let new = old + value;
        self.write_u32(addr, new.to_bits())?;
        Ok(old)
    }
}

#[derive(Debug)]
pub struct LocalMemory {
    data: Vec<u8>,
}

impl LocalMemory {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0; size],
        }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    fn check_bounds(&self, addr: u32, size: usize) -> Result<usize, EmulatorError> {
        let addr = addr as usize;
        if addr
            .checked_add(size)
            .is_none_or(|end| end > self.data.len())
        {
            return Err(EmulatorError::MemoryOutOfBounds {
                address: addr as u64,
            });
        }
        Ok(addr)
    }

    pub fn read_u8(&self, addr: u32) -> Result<u8, EmulatorError> {
        let addr = self.check_bounds(addr, 1)?;
        Ok(self.data[addr])
    }

    pub fn read_u16(&self, addr: u32) -> Result<u16, EmulatorError> {
        let addr = self.check_bounds(addr, 2)?;
        Ok(u16::from_le_bytes([self.data[addr], self.data[addr + 1]]))
    }

    pub fn read_u32(&self, addr: u32) -> Result<u32, EmulatorError> {
        let addr = self.check_bounds(addr, 4)?;
        Ok(u32::from_le_bytes([
            self.data[addr],
            self.data[addr + 1],
            self.data[addr + 2],
            self.data[addr + 3],
        ]))
    }

    pub fn read_u64(&self, addr: u32) -> Result<u64, EmulatorError> {
        let addr = self.check_bounds(addr, 8)?;
        Ok(u64::from_le_bytes([
            self.data[addr],
            self.data[addr + 1],
            self.data[addr + 2],
            self.data[addr + 3],
            self.data[addr + 4],
            self.data[addr + 5],
            self.data[addr + 6],
            self.data[addr + 7],
        ]))
    }

    pub fn write_u8(&mut self, addr: u32, value: u8) -> Result<(), EmulatorError> {
        let addr = self.check_bounds(addr, 1)?;
        self.data[addr] = value;
        Ok(())
    }

    pub fn write_u16(&mut self, addr: u32, value: u16) -> Result<(), EmulatorError> {
        let addr = self.check_bounds(addr, 2)?;
        let bytes = value.to_le_bytes();
        self.data[addr..addr + 2].copy_from_slice(&bytes);
        Ok(())
    }

    pub fn write_u32(&mut self, addr: u32, value: u32) -> Result<(), EmulatorError> {
        let addr = self.check_bounds(addr, 4)?;
        let bytes = value.to_le_bytes();
        self.data[addr..addr + 4].copy_from_slice(&bytes);
        Ok(())
    }

    pub fn write_u64(&mut self, addr: u32, value: u64) -> Result<(), EmulatorError> {
        let addr = self.check_bounds(addr, 8)?;
        let bytes = value.to_le_bytes();
        self.data[addr..addr + 8].copy_from_slice(&bytes);
        Ok(())
    }

    pub fn atomic_add(&mut self, addr: u32, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old.wrapping_add(value))?;
        Ok(old)
    }

    pub fn atomic_sub(&mut self, addr: u32, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old.wrapping_sub(value))?;
        Ok(old)
    }

    pub fn atomic_min(&mut self, addr: u32, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old.min(value))?;
        Ok(old)
    }

    pub fn atomic_max(&mut self, addr: u32, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old.max(value))?;
        Ok(old)
    }

    pub fn atomic_and(&mut self, addr: u32, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old & value)?;
        Ok(old)
    }

    pub fn atomic_or(&mut self, addr: u32, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old | value)?;
        Ok(old)
    }

    pub fn atomic_xor(&mut self, addr: u32, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old ^ value)?;
        Ok(old)
    }

    pub fn atomic_exchange(&mut self, addr: u32, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, value)?;
        Ok(old)
    }

    pub fn atomic_cas(
        &mut self,
        addr: u32,
        expected: u32,
        desired: u32,
    ) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        if old == expected {
            self.write_u32(addr, desired)?;
        }
        Ok(old)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_memory_read_write_u32() {
        let mut mem = DeviceMemory::new(1024);
        mem.write_u32(0x100, 0xDEADBEEF).unwrap();
        assert_eq!(mem.read_u32(0x100).unwrap(), 0xDEADBEEF);
    }

    #[test]
    fn test_device_memory_read_write_u64() {
        let mut mem = DeviceMemory::new(1024);
        mem.write_u64(0x100, 0x123456789ABCDEF0).unwrap();
        assert_eq!(mem.read_u64(0x100).unwrap(), 0x123456789ABCDEF0);
    }

    #[test]
    fn test_device_memory_bounds_check() {
        let mem = DeviceMemory::new(1024);
        assert!(mem.read_u32(1024).is_err());
        assert!(mem.read_u32(1021).is_err());
    }

    #[test]
    fn test_device_memory_atomic_add() {
        let mut mem = DeviceMemory::new(1024);
        mem.write_u32(0x100, 10).unwrap();

        let old = mem.atomic_add(0x100, 5).unwrap();
        assert_eq!(old, 10);
        assert_eq!(mem.read_u32(0x100).unwrap(), 15);
    }

    #[test]
    fn test_device_memory_atomic_cas() {
        let mut mem = DeviceMemory::new(1024);
        mem.write_u32(0x100, 42).unwrap();

        let old = mem.atomic_cas(0x100, 42, 100).unwrap();
        assert_eq!(old, 42);
        assert_eq!(mem.read_u32(0x100).unwrap(), 100);

        let old = mem.atomic_cas(0x100, 42, 200).unwrap();
        assert_eq!(old, 100);
        assert_eq!(mem.read_u32(0x100).unwrap(), 100);
    }

    #[test]
    fn test_local_memory_read_write() {
        let mut mem = LocalMemory::new(1024);
        mem.write_u32(0x100, 0xCAFEBABE).unwrap();
        assert_eq!(mem.read_u32(0x100).unwrap(), 0xCAFEBABE);
    }

    #[test]
    fn test_local_memory_atomic_ops() {
        let mut mem = LocalMemory::new(1024);
        mem.write_u32(0, 100).unwrap();

        let old = mem.atomic_sub(0, 30).unwrap();
        assert_eq!(old, 100);
        assert_eq!(mem.read_u32(0).unwrap(), 70);

        let old = mem.atomic_min(0, 50).unwrap();
        assert_eq!(old, 70);
        assert_eq!(mem.read_u32(0).unwrap(), 50);
    }
}
