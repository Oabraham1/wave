// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Type system for WAVE GPU kernels.
//!
//! All types must be representable in WAVE registers. The type system
//! covers scalar types, pointers with address spaces, fixed-size arrays,
//! and void for functions with no return value.

/// Memory address spaces for GPU memory hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddressSpace {
    /// Per-thread private memory (registers/stack).
    Private,
    /// Workgroup shared memory (scratchpad).
    Local,
    /// Global device memory.
    Device,
}

/// GPU-specific type system where all types map to WAVE registers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
    /// 32-bit floating point.
    F32,
    /// 16-bit floating point (stored in lower half of 32-bit register).
    F16,
    /// 64-bit floating point (register pair).
    F64,
    /// Boolean predicate (maps to WAVE predicate register).
    Bool,
    /// Pointer to memory in a specific address space.
    Ptr(AddressSpace),
    /// Fixed-size array of a given element type.
    Array(Box<Type>, usize),
    /// No return value.
    Void,
}

impl Type {
    /// Returns the size in bytes of this type.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::U32 | Self::I32 | Self::F32 | Self::Bool | Self::Ptr(_) => 4,
            Self::F16 => 2,
            Self::F64 => 8,
            Self::Array(elem, count) => elem.size_bytes() * count,
            Self::Void => 0,
        }
    }

    /// Returns true if this is a floating-point type.
    #[must_use]
    pub fn is_float(&self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::F64)
    }

    /// Returns true if this is an integer type.
    #[must_use]
    pub fn is_integer(&self) -> bool {
        matches!(self, Self::U32 | Self::I32)
    }

    /// Returns true if this is a pointer type.
    #[must_use]
    pub fn is_pointer(&self) -> bool {
        matches!(self, Self::Ptr(_))
    }

    /// Returns the number of 32-bit registers needed to hold this type.
    #[must_use]
    pub fn register_count(&self) -> usize {
        match self {
            Self::U32 | Self::I32 | Self::F32 | Self::F16 | Self::Bool | Self::Ptr(_) => 1,
            Self::F64 => 2,
            Self::Array(elem, count) => elem.register_count() * count,
            Self::Void => 0,
        }
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::U32 => write!(f, "u32"),
            Self::I32 => write!(f, "i32"),
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::F64 => write!(f, "f64"),
            Self::Bool => write!(f, "bool"),
            Self::Ptr(space) => write!(f, "ptr<{space:?}>"),
            Self::Array(elem, size) => write!(f, "[{elem}; {size}]"),
            Self::Void => write!(f, "void"),
        }
    }
}

impl std::fmt::Display for AddressSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Private => write!(f, "private"),
            Self::Local => write!(f, "local"),
            Self::Device => write!(f, "device"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_sizes() {
        assert_eq!(Type::U32.size_bytes(), 4);
        assert_eq!(Type::I32.size_bytes(), 4);
        assert_eq!(Type::F32.size_bytes(), 4);
        assert_eq!(Type::F16.size_bytes(), 2);
        assert_eq!(Type::F64.size_bytes(), 8);
        assert_eq!(Type::Bool.size_bytes(), 4);
        assert_eq!(Type::Ptr(AddressSpace::Device).size_bytes(), 4);
        assert_eq!(Type::Array(Box::new(Type::F32), 4).size_bytes(), 16);
        assert_eq!(Type::Void.size_bytes(), 0);
    }

    #[test]
    fn test_type_classification() {
        assert!(Type::F32.is_float());
        assert!(Type::F16.is_float());
        assert!(Type::F64.is_float());
        assert!(!Type::U32.is_float());
        assert!(Type::U32.is_integer());
        assert!(Type::I32.is_integer());
        assert!(!Type::F32.is_integer());
        assert!(Type::Ptr(AddressSpace::Device).is_pointer());
        assert!(!Type::U32.is_pointer());
    }

    #[test]
    fn test_register_count() {
        assert_eq!(Type::U32.register_count(), 1);
        assert_eq!(Type::F64.register_count(), 2);
        assert_eq!(Type::Array(Box::new(Type::F32), 4).register_count(), 4);
        assert_eq!(Type::Void.register_count(), 0);
    }

    #[test]
    fn test_type_display() {
        assert_eq!(format!("{}", Type::U32), "u32");
        assert_eq!(
            format!("{}", Type::Ptr(AddressSpace::Device)),
            "ptr<Device>"
        );
        assert_eq!(
            format!("{}", Type::Array(Box::new(Type::F32), 4)),
            "[f32; 4]"
        );
    }
}
