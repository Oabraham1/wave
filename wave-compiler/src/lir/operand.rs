// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! LIR operand types: virtual registers, physical registers, and immediates.
//!
//! Operands represent the inputs and outputs of near-WAVE instructions.
//! Before register allocation, virtual registers are used. After allocation,
//! physical registers replace them.

/// Virtual register identifier (before register allocation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VReg(pub u32);

impl std::fmt::Display for VReg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// Physical register identifier (after register allocation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PhysReg(pub u8);

impl std::fmt::Display for PhysReg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "r{}", self.0)
    }
}

/// Predicate register (p0-p3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PReg(pub u8);

impl std::fmt::Display for PReg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "p{}", self.0)
    }
}

/// Special register index for GPU intrinsics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecialReg {
    /// Thread ID X (sr0).
    ThreadIdX,
    /// Thread ID Y (sr1).
    ThreadIdY,
    /// Thread ID Z (sr2).
    ThreadIdZ,
    /// Wave ID (sr3).
    WaveId,
    /// Lane ID (sr4).
    LaneId,
    /// Workgroup ID X (sr5).
    WorkgroupIdX,
    /// Workgroup ID Y (sr6).
    WorkgroupIdY,
    /// Workgroup ID Z (sr7).
    WorkgroupIdZ,
    /// Workgroup size X (sr8).
    WorkgroupSizeX,
    /// Workgroup size Y (sr9).
    WorkgroupSizeY,
    /// Workgroup size Z (sr10).
    WorkgroupSizeZ,
    /// Wave width (sr14).
    WaveWidth,
}

impl SpecialReg {
    /// Returns the special register index.
    #[must_use]
    pub fn index(self) -> u8 {
        match self {
            Self::ThreadIdX => 0,
            Self::ThreadIdY => 1,
            Self::ThreadIdZ => 2,
            Self::WaveId => 3,
            Self::LaneId => 4,
            Self::WorkgroupIdX => 5,
            Self::WorkgroupIdY => 6,
            Self::WorkgroupIdZ => 7,
            Self::WorkgroupSizeX => 8,
            Self::WorkgroupSizeY => 9,
            Self::WorkgroupSizeZ => 10,
            Self::WaveWidth => 14,
        }
    }
}

/// Memory access width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemWidth {
    /// 8-bit access.
    W8,
    /// 16-bit access.
    W16,
    /// 32-bit access.
    W32,
    /// 64-bit access.
    W64,
}

impl MemWidth {
    /// Returns the width in bytes.
    #[must_use]
    pub fn bytes(self) -> u32 {
        match self {
            Self::W8 => 1,
            Self::W16 => 2,
            Self::W32 => 4,
            Self::W64 => 8,
        }
    }

    /// Returns the wave-decode `MemWidth` encoding value.
    #[must_use]
    pub fn encoding(self) -> u8 {
        match self {
            Self::W8 => 0,
            Self::W16 => 1,
            Self::W32 => 2,
            Self::W64 => 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vreg_display() {
        assert_eq!(format!("{}", VReg(0)), "v0");
        assert_eq!(format!("{}", VReg(42)), "v42");
    }

    #[test]
    fn test_physreg_display() {
        assert_eq!(format!("{}", PhysReg(5)), "r5");
    }

    #[test]
    fn test_special_reg_indices() {
        assert_eq!(SpecialReg::ThreadIdX.index(), 0);
        assert_eq!(SpecialReg::LaneId.index(), 4);
        assert_eq!(SpecialReg::WaveWidth.index(), 14);
    }

    #[test]
    fn test_mem_width_bytes_and_encoding() {
        assert_eq!(MemWidth::W8.bytes(), 1);
        assert_eq!(MemWidth::W16.bytes(), 2);
        assert_eq!(MemWidth::W32.bytes(), 4);
        assert_eq!(MemWidth::W64.bytes(), 8);
        assert_eq!(MemWidth::W8.encoding(), 0);
        assert_eq!(MemWidth::W32.encoding(), 2);
    }
}
