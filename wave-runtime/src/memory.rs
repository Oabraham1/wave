// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Host buffers for kernel arguments and results. Vendor launches exchange buffer contents
//! through temporary files.

use crate::error::RuntimeError;
use std::fmt;

/// Element data type for buffer contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    /// 32-bit float.
    F32,
    /// 32-bit unsigned integer.
    U32,
    /// 32-bit signed integer.
    I32,
    /// 16-bit float.
    F16,
    /// 64-bit float.
    F64,
}

impl ElementType {
    /// Size of one element in bytes.
    #[must_use]
    pub fn size_bytes(self) -> usize {
        match self {
            Self::F32 | Self::U32 | Self::I32 => 4,
            Self::F16 => 2,
            Self::F64 => 8,
        }
    }
}

impl fmt::Display for ElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::U32 => write!(f, "u32"),
            Self::I32 => write!(f, "i32"),
            Self::F16 => write!(f, "f16"),
            Self::F64 => write!(f, "f64"),
        }
    }
}

/// CPU-side buffer representing device memory.
#[derive(Debug, Clone)]
pub struct DeviceBuffer {
    /// Raw bytes of the buffer.
    pub data: Vec<u8>,
    /// Number of elements in the buffer.
    pub count: usize,
    /// Element data type.
    pub element_type: ElementType,
}

impl DeviceBuffer {
    /// Create a buffer from an `f32` slice.
    #[must_use]
    pub fn from_f32(data: &[f32]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            data: bytes,
            count: data.len(),
            element_type: ElementType::F32,
        }
    }

    /// Create a zero-filled `f32` buffer.
    #[must_use]
    pub fn zeros_f32(count: usize) -> Self {
        Self {
            data: vec![0u8; count * 4],
            count,
            element_type: ElementType::F32,
        }
    }

    /// Create a buffer from a `u32` slice.
    #[must_use]
    pub fn from_u32(data: &[u32]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            data: bytes,
            count: data.len(),
            element_type: ElementType::U32,
        }
    }

    /// Create a zero-filled `u32` buffer.
    #[must_use]
    pub fn zeros_u32(count: usize) -> Self {
        Self {
            data: vec![0u8; count * 4],
            count,
            element_type: ElementType::U32,
        }
    }

    /// Create a buffer from an `i32` slice.
    #[must_use]
    pub fn from_i32(data: &[i32]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            data: bytes,
            count: data.len(),
            element_type: ElementType::I32,
        }
    }

    /// Read buffer contents as `f32` values.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError::Memory` for a different element type or invalid byte length.
    pub fn to_f32(&self) -> Result<Vec<f32>, RuntimeError> {
        Ok(self.iter_f32()?.collect())
    }

    /// Iterate over the buffer's little-endian `f32` values without copying them.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError::Memory` for a different element type or invalid byte length.
    ///
    /// # Examples
    ///
    /// ```
    /// use wave_runtime::memory::DeviceBuffer;
    /// let buffer = DeviceBuffer::from_f32(&[1.0, 2.0]);
    /// assert_eq!(buffer.iter_f32()?.sum::<f32>(), 3.0);
    /// # Ok::<(), wave_runtime::error::RuntimeError>(())
    /// ```
    pub fn iter_f32(&self) -> Result<impl ExactSizeIterator<Item = f32> + '_, RuntimeError> {
        if self.element_type != ElementType::F32 {
            return Err(RuntimeError::Memory(format!(
                "cannot read {} buffer as f32",
                self.element_type
            )));
        }
        self.validate()?;
        Ok(self
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(f32::from_le_bytes))
    }

    /// Check that element count and byte length agree.
    pub(crate) fn validate(&self) -> Result<(), RuntimeError> {
        if self.count.checked_mul(self.element_type.size_bytes()) != Some(self.data.len()) {
            return Err(RuntimeError::Memory(format!(
                "{} elements of {} do not match {} bytes",
                self.count,
                self.element_type,
                self.data.len()
            )));
        }
        Ok(())
    }

    /// Read buffer contents as `u32` values.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError::Memory` for a different element type or invalid byte length.
    pub fn to_u32(&self) -> Result<Vec<u32>, RuntimeError> {
        if self.element_type != ElementType::U32 {
            return Err(RuntimeError::Memory(format!(
                "cannot read {} buffer as u32",
                self.element_type
            )));
        }
        self.validate()?;
        Ok(self
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Total size of the buffer in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_f32_roundtrip() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let buf = DeviceBuffer::from_f32(&data);
        assert_eq!(buf.count, 4);
        assert_eq!(buf.element_type, ElementType::F32);
        assert_eq!(buf.to_f32().unwrap(), data);
    }

    #[test]
    fn test_zeros_f32() {
        let buf = DeviceBuffer::zeros_f32(8);
        assert_eq!(buf.count, 8);
        assert_eq!(buf.size_bytes(), 32);
        assert_eq!(buf.to_f32().unwrap(), vec![0.0; 8]);
    }

    #[test]
    fn test_from_u32_roundtrip() {
        let data = vec![10_u32, 20, 30];
        let buf = DeviceBuffer::from_u32(&data);
        assert_eq!(buf.to_u32().unwrap(), data);
    }

    #[test]
    fn test_type_mismatch() {
        let buf = DeviceBuffer::from_u32(&[1, 2]);
        assert!(buf.to_f32().is_err());
    }
    #[test]
    fn typed_reads_reject_invalid_lengths() {
        let mut buffer = DeviceBuffer::from_f32(&[1.0, 2.0]);
        buffer.data.push(0);
        assert!(buffer.iter_f32().is_err());
        assert!(buffer.to_f32().is_err());
        buffer.element_type = ElementType::U32;
        assert!(buffer.to_u32().is_err());
        buffer.count = usize::MAX;
        assert!(buffer.to_u32().is_err());
    }
}
