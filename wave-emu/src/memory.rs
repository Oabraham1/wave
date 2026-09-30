// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Bounds-checked device and workgroup-local memory with loads and stores at each supported
//! width.

use crate::EmulatorError;

#[derive(Debug)]
pub struct DeviceMemory {
    data: Vec<u8>,
}

impl DeviceMemory {
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0; size],
        }
    }

    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    fn check_bounds(&self, addr: u64, size: usize) -> Result<usize, EmulatorError> {
        let address = addr;
        let addr =
            usize::try_from(addr).map_err(|_| EmulatorError::MemoryOutOfBounds { address })?;
        if addr
            .checked_add(size)
            .is_none_or(|end| end > self.data.len())
        {
            return Err(EmulatorError::MemoryOutOfBounds { address });
        }
        Ok(addr)
    }

    /// Read a little-endian u8 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use wave_emu::memory::DeviceMemory;
    /// let mut memory = DeviceMemory::new(16);
    /// memory.write_u8(0, 42)?;
    /// assert_eq!(memory.read_u8(0)?, 42);
    /// # Ok::<(), wave_emu::EmulatorError>(())
    /// ```
    pub fn read_u8(&self, addr: u64) -> Result<u8, EmulatorError> {
        let addr = self.check_bounds(addr, 1)?;
        Ok(self.data[addr])
    }

    /// Read a little-endian u16 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use wave_emu::memory::DeviceMemory;
    /// let mut memory = DeviceMemory::new(16);
    /// memory.write_u16(0, 42)?;
    /// assert_eq!(memory.read_u16(0)?, 42);
    /// # Ok::<(), wave_emu::EmulatorError>(())
    /// ```
    pub fn read_u16(&self, addr: u64) -> Result<u16, EmulatorError> {
        let addr = self.check_bounds(addr, 2)?;
        Ok(u16::from_le_bytes([self.data[addr], self.data[addr + 1]]))
    }

    /// Read a little-endian u32 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use wave_emu::memory::DeviceMemory;
    /// let mut memory = DeviceMemory::new(16);
    /// memory.write_u32(0, 42)?;
    /// assert_eq!(memory.read_u32(0)?, 42);
    /// # Ok::<(), wave_emu::EmulatorError>(())
    /// ```
    pub fn read_u32(&self, addr: u64) -> Result<u32, EmulatorError> {
        let addr = self.check_bounds(addr, 4)?;
        Ok(u32::from_le_bytes([
            self.data[addr],
            self.data[addr + 1],
            self.data[addr + 2],
            self.data[addr + 3],
        ]))
    }

    /// Read a little-endian u64 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use wave_emu::memory::DeviceMemory;
    /// let mut memory = DeviceMemory::new(16);
    /// memory.write_u64(0, 42)?;
    /// assert_eq!(memory.read_u64(0)?, 42);
    /// # Ok::<(), wave_emu::EmulatorError>(())
    /// ```
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

    /// Read a little-endian u128 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use wave_emu::memory::DeviceMemory;
    /// let mut memory = DeviceMemory::new(16);
    /// memory.write_u128(0, 42)?;
    /// assert_eq!(memory.read_u128(0)?, 42);
    /// # Ok::<(), wave_emu::EmulatorError>(())
    /// ```
    pub fn read_u128(&self, addr: u64) -> Result<u128, EmulatorError> {
        let addr = self.check_bounds(addr, 16)?;
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&self.data[addr..addr + 16]);
        Ok(u128::from_le_bytes(bytes))
    }

    /// Write a little-endian u8 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use wave_emu::memory::DeviceMemory;
    /// let mut memory = DeviceMemory::new(16);
    /// memory.write_u8(0, 42)?;
    /// assert_eq!(memory.read_u8(0)?, 42);
    /// # Ok::<(), wave_emu::EmulatorError>(())
    /// ```
    pub fn write_u8(&mut self, addr: u64, value: u8) -> Result<(), EmulatorError> {
        let addr = self.check_bounds(addr, 1)?;
        self.data[addr] = value;
        Ok(())
    }

    /// Write a little-endian u16 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use wave_emu::memory::DeviceMemory;
    /// let mut memory = DeviceMemory::new(16);
    /// memory.write_u16(0, 42)?;
    /// assert_eq!(memory.read_u16(0)?, 42);
    /// # Ok::<(), wave_emu::EmulatorError>(())
    /// ```
    pub fn write_u16(&mut self, addr: u64, value: u16) -> Result<(), EmulatorError> {
        let addr = self.check_bounds(addr, 2)?;
        let bytes = value.to_le_bytes();
        self.data[addr..addr + 2].copy_from_slice(&bytes);
        Ok(())
    }

    /// Write a little-endian u32 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use wave_emu::memory::DeviceMemory;
    /// let mut memory = DeviceMemory::new(16);
    /// memory.write_u32(0, 42)?;
    /// assert_eq!(memory.read_u32(0)?, 42);
    /// # Ok::<(), wave_emu::EmulatorError>(())
    /// ```
    pub fn write_u32(&mut self, addr: u64, value: u32) -> Result<(), EmulatorError> {
        let addr = self.check_bounds(addr, 4)?;
        let bytes = value.to_le_bytes();
        self.data[addr..addr + 4].copy_from_slice(&bytes);
        Ok(())
    }

    /// Write a little-endian u64 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use wave_emu::memory::DeviceMemory;
    /// let mut memory = DeviceMemory::new(16);
    /// memory.write_u64(0, 42)?;
    /// assert_eq!(memory.read_u64(0)?, 42);
    /// # Ok::<(), wave_emu::EmulatorError>(())
    /// ```
    pub fn write_u64(&mut self, addr: u64, value: u64) -> Result<(), EmulatorError> {
        let addr = self.check_bounds(addr, 8)?;
        let bytes = value.to_le_bytes();
        self.data[addr..addr + 8].copy_from_slice(&bytes);
        Ok(())
    }

    /// Write a little-endian u128 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use wave_emu::memory::DeviceMemory;
    /// let mut memory = DeviceMemory::new(16);
    /// memory.write_u128(0, 42)?;
    /// assert_eq!(memory.read_u128(0)?, 42);
    /// # Ok::<(), wave_emu::EmulatorError>(())
    /// ```
    pub fn write_u128(&mut self, addr: u64, value: u128) -> Result<(), EmulatorError> {
        let addr = self.check_bounds(addr, 16)?;
        let bytes = value.to_le_bytes();
        self.data[addr..addr + 16].copy_from_slice(&bytes);
        Ok(())
    }

    /// Borrow a contiguous range of device memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the range is outside device memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use wave_emu::memory::DeviceMemory;
    /// let memory = DeviceMemory::new(8);
    /// assert_eq!(memory.read_slice(4, 4)?, &[0; 4]);
    /// assert!(memory.read_slice(u64::MAX, 2).is_err());
    /// # Ok::<(), wave_emu::EmulatorError>(())
    /// ```
    pub fn read_slice(&self, offset: u64, len: usize) -> Result<&[u8], EmulatorError> {
        let start = self.check_bounds(offset, len)?;
        Ok(&self.data[start..start + len])
    }

    /// Copy bytes into memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    pub fn write_slice(&mut self, offset: u64, data: &[u8]) -> Result<(), EmulatorError> {
        let addr = self.check_bounds(offset, data.len())?;
        self.data[addr..addr + data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Apply add and return the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    pub fn atomic_add(&mut self, addr: u64, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old.wrapping_add(value))?;
        Ok(old)
    }

    /// Apply sub and return the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    pub fn atomic_sub(&mut self, addr: u64, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old.wrapping_sub(value))?;
        Ok(old)
    }

    /// Apply min and return the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    pub fn atomic_min(&mut self, addr: u64, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old.min(value))?;
        Ok(old)
    }

    /// Apply max and return the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    pub fn atomic_max(&mut self, addr: u64, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old.max(value))?;
        Ok(old)
    }

    /// Apply and and return the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    pub fn atomic_and(&mut self, addr: u64, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old & value)?;
        Ok(old)
    }

    /// Apply or and return the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    pub fn atomic_or(&mut self, addr: u64, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old | value)?;
        Ok(old)
    }

    /// Apply xor and return the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    pub fn atomic_xor(&mut self, addr: u64, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old ^ value)?;
        Ok(old)
    }

    /// Apply exchange and return the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    pub fn atomic_exchange(&mut self, addr: u64, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, value)?;
        Ok(old)
    }

    /// Apply cas and return the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
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

    /// Apply add i32 and return the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    pub fn atomic_add_i32(&mut self, addr: u64, value: i32) -> Result<i32, EmulatorError> {
        let old = self.read_u32(addr)?.cast_signed();
        self.write_u32(addr, old.wrapping_add(value).cast_unsigned())?;
        Ok(old)
    }

    /// Apply min i32 and return the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    pub fn atomic_min_i32(&mut self, addr: u64, value: i32) -> Result<i32, EmulatorError> {
        let old = self.read_u32(addr)?.cast_signed();
        self.write_u32(addr, old.min(value).cast_unsigned())?;
        Ok(old)
    }

    /// Apply max i32 and return the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    pub fn atomic_max_i32(&mut self, addr: u64, value: i32) -> Result<i32, EmulatorError> {
        let old = self.read_u32(addr)?.cast_signed();
        self.write_u32(addr, old.max(value).cast_unsigned())?;
        Ok(old)
    }

    /// Apply add f32 and return the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
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
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0; size],
        }
    }

    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    fn check_bounds(&self, addr: u32, size: usize) -> Result<usize, EmulatorError> {
        let address = addr;
        let addr = usize::try_from(addr).map_err(|_| EmulatorError::MemoryOutOfBounds {
            address: u64::from(address),
        })?;
        if addr
            .checked_add(size)
            .is_none_or(|end| end > self.data.len())
        {
            return Err(EmulatorError::MemoryOutOfBounds {
                address: u64::from(address),
            });
        }
        Ok(addr)
    }

    /// Read a little-endian u8 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use wave_emu::memory::LocalMemory;
    /// let mut memory = LocalMemory::new(16);
    /// memory.write_u8(0, 42)?;
    /// assert_eq!(memory.read_u8(0)?, 42);
    /// # Ok::<(), wave_emu::EmulatorError>(())
    /// ```
    pub fn read_u8(&self, addr: u32) -> Result<u8, EmulatorError> {
        let addr = self.check_bounds(addr, 1)?;
        Ok(self.data[addr])
    }

    /// Read a little-endian u16 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use wave_emu::memory::LocalMemory;
    /// let mut memory = LocalMemory::new(16);
    /// memory.write_u16(0, 42)?;
    /// assert_eq!(memory.read_u16(0)?, 42);
    /// # Ok::<(), wave_emu::EmulatorError>(())
    /// ```
    pub fn read_u16(&self, addr: u32) -> Result<u16, EmulatorError> {
        let addr = self.check_bounds(addr, 2)?;
        Ok(u16::from_le_bytes([self.data[addr], self.data[addr + 1]]))
    }

    /// Read a little-endian u32 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use wave_emu::memory::LocalMemory;
    /// let mut memory = LocalMemory::new(16);
    /// memory.write_u32(0, 42)?;
    /// assert_eq!(memory.read_u32(0)?, 42);
    /// # Ok::<(), wave_emu::EmulatorError>(())
    /// ```
    pub fn read_u32(&self, addr: u32) -> Result<u32, EmulatorError> {
        let addr = self.check_bounds(addr, 4)?;
        Ok(u32::from_le_bytes([
            self.data[addr],
            self.data[addr + 1],
            self.data[addr + 2],
            self.data[addr + 3],
        ]))
    }

    /// Read a little-endian u64 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use wave_emu::memory::LocalMemory;
    /// let mut memory = LocalMemory::new(16);
    /// memory.write_u64(0, 42)?;
    /// assert_eq!(memory.read_u64(0)?, 42);
    /// # Ok::<(), wave_emu::EmulatorError>(())
    /// ```
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

    /// Write a little-endian u8 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use wave_emu::memory::LocalMemory;
    /// let mut memory = LocalMemory::new(16);
    /// memory.write_u8(0, 42)?;
    /// assert_eq!(memory.read_u8(0)?, 42);
    /// # Ok::<(), wave_emu::EmulatorError>(())
    /// ```
    pub fn write_u8(&mut self, addr: u32, value: u8) -> Result<(), EmulatorError> {
        let addr = self.check_bounds(addr, 1)?;
        self.data[addr] = value;
        Ok(())
    }

    /// Write a little-endian u16 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use wave_emu::memory::LocalMemory;
    /// let mut memory = LocalMemory::new(16);
    /// memory.write_u16(0, 42)?;
    /// assert_eq!(memory.read_u16(0)?, 42);
    /// # Ok::<(), wave_emu::EmulatorError>(())
    /// ```
    pub fn write_u16(&mut self, addr: u32, value: u16) -> Result<(), EmulatorError> {
        let addr = self.check_bounds(addr, 2)?;
        let bytes = value.to_le_bytes();
        self.data[addr..addr + 2].copy_from_slice(&bytes);
        Ok(())
    }

    /// Write a little-endian u32 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use wave_emu::memory::LocalMemory;
    /// let mut memory = LocalMemory::new(16);
    /// memory.write_u32(0, 42)?;
    /// assert_eq!(memory.read_u32(0)?, 42);
    /// # Ok::<(), wave_emu::EmulatorError>(())
    /// ```
    pub fn write_u32(&mut self, addr: u32, value: u32) -> Result<(), EmulatorError> {
        let addr = self.check_bounds(addr, 4)?;
        let bytes = value.to_le_bytes();
        self.data[addr..addr + 4].copy_from_slice(&bytes);
        Ok(())
    }

    /// Write a little-endian u64 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use wave_emu::memory::LocalMemory;
    /// let mut memory = LocalMemory::new(16);
    /// memory.write_u64(0, 42)?;
    /// assert_eq!(memory.read_u64(0)?, 42);
    /// # Ok::<(), wave_emu::EmulatorError>(())
    /// ```
    pub fn write_u64(&mut self, addr: u32, value: u64) -> Result<(), EmulatorError> {
        let addr = self.check_bounds(addr, 8)?;
        let bytes = value.to_le_bytes();
        self.data[addr..addr + 8].copy_from_slice(&bytes);
        Ok(())
    }

    /// Apply add and return the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    pub fn atomic_add(&mut self, addr: u32, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old.wrapping_add(value))?;
        Ok(old)
    }

    /// Apply sub and return the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    pub fn atomic_sub(&mut self, addr: u32, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old.wrapping_sub(value))?;
        Ok(old)
    }

    /// Apply min and return the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    pub fn atomic_min(&mut self, addr: u32, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old.min(value))?;
        Ok(old)
    }

    /// Apply max and return the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    pub fn atomic_max(&mut self, addr: u32, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old.max(value))?;
        Ok(old)
    }

    /// Apply and and return the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    pub fn atomic_and(&mut self, addr: u32, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old & value)?;
        Ok(old)
    }

    /// Apply or and return the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    pub fn atomic_or(&mut self, addr: u32, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old | value)?;
        Ok(old)
    }

    /// Apply xor and return the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    pub fn atomic_xor(&mut self, addr: u32, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, old ^ value)?;
        Ok(old)
    }

    /// Apply exchange and return the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
    pub fn atomic_exchange(&mut self, addr: u32, value: u32) -> Result<u32, EmulatorError> {
        let old = self.read_u32(addr)?;
        self.write_u32(addr, value)?;
        Ok(old)
    }

    /// Apply cas and return the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the accessed range is outside memory.
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
        mem.write_u32(0x100, 0xDEAD_BEEF).unwrap();
        assert_eq!(mem.read_u32(0x100).unwrap(), 0xDEAD_BEEF);
    }

    #[test]
    fn test_device_memory_read_write_u64() {
        let mut mem = DeviceMemory::new(1024);
        mem.write_u64(0x100, 0x1234_5678_9ABC_DEF0).unwrap();
        assert_eq!(mem.read_u64(0x100).unwrap(), 0x1234_5678_9ABC_DEF0);
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
        mem.write_u32(0x100, 0xCAFE_BABE).unwrap();
        assert_eq!(mem.read_u32(0x100).unwrap(), 0xCAFE_BABE);
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
