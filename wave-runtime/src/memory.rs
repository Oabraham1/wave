// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Device memory buffer types for the WAVE runtime.
//!
//! Provides CPU-side buffer management for kernel arguments. Buffers are
//! serialized to temporary files for subprocess-based kernel launch in v1.
//! Direct GPU memory mapping is planned for v2.

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
    /// Returns `RuntimeError::Memory` if the buffer element type is not `f32`.
    pub fn to_f32(&self) -> Result<Vec<f32>, RuntimeError> {
        if self.element_type != ElementType::F32 {
            return Err(RuntimeError::Memory(format!(
                "cannot read {} buffer as f32",
                self.element_type
            )));
        }
        Ok(self
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Read buffer contents as `u32` values.
    ///
    /// # Errors
    ///
    /// Returns `RuntimeError::Memory` if the buffer element type is not `u32`.
    pub fn to_u32(&self) -> Result<Vec<u32>, RuntimeError> {
        if self.element_type != ElementType::U32 {
            return Err(RuntimeError::Memory(format!(
                "cannot read {} buffer as u32",
                self.element_type
            )));
        }
        Ok(self
            .data
            .chunks_exact(4)
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
}
