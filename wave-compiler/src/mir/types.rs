// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! MIR type system, lowered from HIR types.
//!
//! MIR types are a simpler representation that directly maps to
//! WAVE register types and memory operations.

use crate::hir;

/// MIR type, a lowered version of HIR types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MirType {
    /// 32-bit integer (signed or unsigned).
    I32,
    /// 32-bit float.
    F32,
    /// 16-bit float.
    F16,
    /// 64-bit float.
    F64,
    /// Boolean predicate.
    Bool,
    /// Pointer (32-bit address).
    Ptr,
}

impl MirType {
    /// Size in bytes.
    #[must_use]
    pub fn size_bytes(self) -> u32 {
        match self {
            Self::I32 | Self::F32 | Self::Ptr | Self::Bool => 4,
            Self::F16 => 2,
            Self::F64 => 8,
        }
    }

    /// Returns true if this is a floating-point type.
    #[must_use]
    pub fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::F64)
    }

    /// Returns true if this is an integer type.
    #[must_use]
    pub fn is_integer(self) -> bool {
        matches!(self, Self::I32 | Self::Ptr)
    }
}

impl std::fmt::Display for MirType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::I32 => write!(f, "i32"),
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::F64 => write!(f, "f64"),
            Self::Bool => write!(f, "bool"),
            Self::Ptr => write!(f, "ptr"),
        }
    }
}

/// Convert an HIR type to a MIR type.
#[must_use]
pub fn lower_type(ty: &hir::Type) -> MirType {
    match ty {
        hir::Type::U32 | hir::Type::I32 | hir::Type::Void => MirType::I32,
        hir::Type::F32 => MirType::F32,
        hir::Type::F16 => MirType::F16,
        hir::Type::F64 => MirType::F64,
        hir::Type::Bool => MirType::Bool,
        hir::Type::Ptr(_) | hir::Type::Array(_, _) => MirType::Ptr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::types::AddressSpace;

    #[test]
    fn test_mir_type_sizes() {
        assert_eq!(MirType::I32.size_bytes(), 4);
        assert_eq!(MirType::F32.size_bytes(), 4);
        assert_eq!(MirType::F16.size_bytes(), 2);
        assert_eq!(MirType::F64.size_bytes(), 8);
        assert_eq!(MirType::Bool.size_bytes(), 4);
        assert_eq!(MirType::Ptr.size_bytes(), 4);
    }

    #[test]
    fn test_lower_type() {
        assert_eq!(lower_type(&hir::Type::U32), MirType::I32);
        assert_eq!(lower_type(&hir::Type::I32), MirType::I32);
        assert_eq!(lower_type(&hir::Type::F32), MirType::F32);
        assert_eq!(lower_type(&hir::Type::F16), MirType::F16);
        assert_eq!(lower_type(&hir::Type::F64), MirType::F64);
        assert_eq!(lower_type(&hir::Type::Bool), MirType::Bool);
        assert_eq!(
            lower_type(&hir::Type::Ptr(AddressSpace::Device)),
            MirType::Ptr
        );
    }

    #[test]
    fn test_type_classification() {
        assert!(MirType::F32.is_float());
        assert!(MirType::F64.is_float());
        assert!(!MirType::I32.is_float());
        assert!(MirType::I32.is_integer());
        assert!(MirType::Ptr.is_integer());
        assert!(!MirType::F32.is_integer());
    }
}
