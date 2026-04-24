// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Decoded instruction types. Represents WAVE instructions in a structured
//!
//! format suitable for disassembly and emulation.

use crate::opcodes::{
    AtomicOp, Bf16Op, Bf16PackedOp, BitOpType, CmpOp, ControlOp, CvtType, F16Op, F16PackedOp,
    F64DivSqrtOp, F64Op, FUnaryOp, MemWidth, MiscOp, MmaOp, Opcode, Scope, SyncOp, WaveOpType,
    WaveReduceType,
};

/// A fully decoded WAVE instruction
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedInstruction {
    /// Byte offset from start of code section
    pub offset: u32,
    /// Size of instruction in bytes (4 or 8)
    pub size: u8,
    /// The decoded operation
    pub operation: Operation,
    /// Predicate register (0 = unpredicated, 1-3 = p1-p3)
    pub predicate: u8,
    /// Whether predicate is negated
    pub predicate_negated: bool,
}

/// All WAVE operations
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    Iadd {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Isub {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Imul {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    ImulHi {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Imad {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rs3: u8,
    },
    Idiv {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Imod {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Ineg {
        rd: u8,
        rs1: u8,
    },
    Iabs {
        rd: u8,
        rs1: u8,
    },
    Imin {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Imax {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Iclamp {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rs3: u8,
    },

    Fadd {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Fsub {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Fmul {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Fma {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rs3: u8,
    },
    Fdiv {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Fneg {
        rd: u8,
        rs1: u8,
    },
    Fabs {
        rd: u8,
        rs1: u8,
    },
    Fmin {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Fmax {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Fclamp {
        rd: u8,
        rs1: u8,
        rs2: u8,
        rs3: u8,
    },
    Fsqrt {
        rd: u8,
        rs1: u8,
    },
    FUnary {
        op: FUnaryOp,
        rd: u8,
        rs1: u8,
    },

    F16 {
        op: F16Op,
        rd: u8,
        rs1: u8,
        rs2: u8,
        rs3: Option<u8>,
    },
    F16Packed {
        op: F16PackedOp,
        rd: u8,
        rs1: u8,
        rs2: u8,
        rs3: Option<u8>,
    },

    Bf16 {
        op: Bf16Op,
        rd: u8,
        rs1: u8,
        rs2: u8,
        rs3: Option<u8>,
    },
    Bf16Packed {
        op: Bf16PackedOp,
        rd: u8,
        rs1: u8,
        rs2: u8,
        rs3: Option<u8>,
    },

    F64 {
        op: F64Op,
        rd: u8,
        rs1: u8,
        rs2: u8,
        rs3: Option<u8>,
    },
    F64DivSqrt {
        op: F64DivSqrtOp,
        rd: u8,
        rs1: u8,
        rs2: Option<u8>,
    },

    And {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Or {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Xor {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Not {
        rd: u8,
        rs1: u8,
    },
    Shl {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Shr {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    Sar {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    BitOp {
        op: BitOpType,
        rd: u8,
        rs1: u8,
        rs2: Option<u8>,
        rs3: Option<u8>,
        rs4: Option<u8>,
    },

    Icmp {
        op: CmpOp,
        pd: u8,
        rs1: u8,
        rs2: u8,
    },
    Ucmp {
        op: CmpOp,
        pd: u8,
        rs1: u8,
        rs2: u8,
    },
    Fcmp {
        op: CmpOp,
        pd: u8,
        rs1: u8,
        rs2: u8,
    },

    Select {
        rd: u8,
        ps: u8,
        rs1: u8,
        rs2: u8,
    },
    Cvt {
        cvt_type: CvtType,
        rd: u8,
        rs1: u8,
    },

    LocalLoad {
        width: MemWidth,
        rd: u8,
        addr: u8,
    },
    LocalStore {
        width: MemWidth,
        addr: u8,
        value: u8,
    },

    DeviceLoad {
        width: MemWidth,
        rd: u8,
        addr: u8,
    },
    DeviceStore {
        width: MemWidth,
        addr: u8,
        value: u8,
    },

    LocalAtomic {
        op: AtomicOp,
        rd: Option<u8>,
        addr: u8,
        value: u8,
    },
    LocalAtomicCas {
        rd: Option<u8>,
        addr: u8,
        expected: u8,
        desired: u8,
    },

    DeviceAtomic {
        op: AtomicOp,
        rd: Option<u8>,
        addr: u8,
        value: u8,
        scope: Scope,
    },
    DeviceAtomicCas {
        rd: Option<u8>,
        addr: u8,
        expected: u8,
        desired: u8,
        scope: Scope,
    },

    WaveOp {
        op: WaveOpType,
        rd: u8,
        rs1: u8,
        rs2: Option<u8>,
    },
    WaveReduce {
        op: WaveReduceType,
        rd: u8,
        rs1: u8,
    },
    WaveBallot {
        rd: u8,
        ps: u8,
    },
    WaveVote {
        op: WaveOpType,
        pd: u8,
        ps: u8,
    },

    If {
        ps: u8,
    },
    Else,
    Endif,
    Loop,
    Break {
        ps: u8,
    },
    Continue {
        ps: u8,
    },
    Endloop,
    Call {
        target: u32,
    },

    Return,
    Halt,
    Barrier,
    FenceAcquire {
        scope: Scope,
    },
    FenceRelease {
        scope: Scope,
    },
    FenceAcqRel {
        scope: Scope,
    },
    Wait,
    Nop,

    Mov {
        rd: u8,
        rs1: u8,
    },
    MovImm {
        rd: u8,
        imm: u32,
    },
    MovSr {
        rd: u8,
        sr_index: u8,
    },

    MmaLoadA {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    MmaLoadB {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    MmaStoreC {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    MmaCompute {
        rd: u8,
        rs1: u8,
        rs2: u8,
    },

    Unknown {
        opcode: u8,
        word0: u32,
        word1: Option<u32>,
    },
}

impl DecodedInstruction {
    /// Returns the mnemonic string for this instruction
    #[must_use]
    pub fn mnemonic(&self) -> String {
        match &self.operation {
            Operation::Iadd { .. } => "iadd".to_string(),
            Operation::Isub { .. } => "isub".to_string(),
            Operation::Imul { .. } => "imul".to_string(),
            Operation::ImulHi { .. } => "imul_hi".to_string(),
            Operation::Imad { .. } => "imad".to_string(),
            Operation::Idiv { .. } => "idiv".to_string(),
            Operation::Imod { .. } => "imod".to_string(),
            Operation::Ineg { .. } => "ineg".to_string(),
            Operation::Iabs { .. } => "iabs".to_string(),
            Operation::Imin { .. } => "imin".to_string(),
            Operation::Imax { .. } => "imax".to_string(),
            Operation::Iclamp { .. } => "iclamp".to_string(),

            Operation::Fadd { .. } => "fadd".to_string(),
            Operation::Fsub { .. } => "fsub".to_string(),
            Operation::Fmul { .. } => "fmul".to_string(),
            Operation::Fma { .. } => "fma".to_string(),
            Operation::Fdiv { .. } => "fdiv".to_string(),
            Operation::Fneg { .. } => "fneg".to_string(),
            Operation::Fabs { .. } => "fabs".to_string(),
            Operation::Fmin { .. } => "fmin".to_string(),
            Operation::Fmax { .. } => "fmax".to_string(),
            Operation::Fclamp { .. } => "fclamp".to_string(),
            Operation::Fsqrt { .. } => "fsqrt".to_string(),
            Operation::FUnary { op, .. } => op.mnemonic().to_string(),

            Operation::F16 { op, .. } => op.mnemonic().to_string(),
            Operation::F16Packed { op, .. } => op.mnemonic().to_string(),
            Operation::Bf16 { op, .. } => op.mnemonic().to_string(),
            Operation::Bf16Packed { op, .. } => op.mnemonic().to_string(),
            Operation::F64 { op, .. } => op.mnemonic().to_string(),
            Operation::F64DivSqrt { op, .. } => op.mnemonic().to_string(),

            Operation::And { .. } => "and".to_string(),
            Operation::Or { .. } => "or".to_string(),
            Operation::Xor { .. } => "xor".to_string(),
            Operation::Not { .. } => "not".to_string(),
            Operation::Shl { .. } => "shl".to_string(),
            Operation::Shr { .. } => "shr".to_string(),
            Operation::Sar { .. } => "sar".to_string(),
            Operation::BitOp { op, .. } => op.mnemonic().to_string(),

            Operation::Icmp { op, .. } => format!("icmp_{}", op.suffix()),
            Operation::Ucmp { op, .. } => format!("ucmp_{}", op.suffix()),
            Operation::Fcmp { op, .. } => format!("fcmp_{}", op.suffix()),

            Operation::Select { .. } => "select".to_string(),
            Operation::Cvt { cvt_type, .. } => cvt_type.mnemonic().to_string(),

            Operation::LocalLoad { width, .. } => format!("local_load_{}", width.suffix()),
            Operation::LocalStore { width, .. } => format!("local_store_{}", width.suffix()),
            Operation::DeviceLoad { width, .. } => format!("device_load_{}", width.suffix()),
            Operation::DeviceStore { width, .. } => format!("device_store_{}", width.suffix()),

            Operation::LocalAtomic { op, .. } => format!("local_atomic_{}", op.suffix()),
            Operation::LocalAtomicCas { .. } => "local_atomic_cas".to_string(),
            Operation::DeviceAtomic { op, .. } => format!("atomic_{}", op.suffix()),
            Operation::DeviceAtomicCas { .. } => "atomic_cas".to_string(),

            Operation::WaveOp { op, .. } | Operation::WaveVote { op, .. } => {
                op.mnemonic().to_string()
            }
            Operation::WaveReduce { op, .. } => op.mnemonic().to_string(),
            Operation::WaveBallot { .. } => "wave_ballot".to_string(),

            Operation::If { .. } => "if".to_string(),
            Operation::Else => "else".to_string(),
            Operation::Endif => "endif".to_string(),
            Operation::Loop => "loop".to_string(),
            Operation::Break { .. } => "break".to_string(),
            Operation::Continue { .. } => "continue".to_string(),
            Operation::Endloop => "endloop".to_string(),
            Operation::Call { .. } => "call".to_string(),

            Operation::Return => "return".to_string(),
            Operation::Halt => "halt".to_string(),
            Operation::Barrier => "barrier".to_string(),
            Operation::FenceAcquire { .. } => "fence_acquire".to_string(),
            Operation::FenceRelease { .. } => "fence_release".to_string(),
            Operation::FenceAcqRel { .. } => "fence_acq_rel".to_string(),
            Operation::Wait => "wait".to_string(),
            Operation::Nop => "nop".to_string(),

            Operation::Mov { .. } | Operation::MovSr { .. } => "mov".to_string(),
            Operation::MovImm { .. } => "mov_imm".to_string(),

            Operation::MmaLoadA { .. } => "mma_load_a".to_string(),
            Operation::MmaLoadB { .. } => "mma_load_b".to_string(),
            Operation::MmaStoreC { .. } => "mma_store_c".to_string(),
            Operation::MmaCompute { .. } => "mma_compute".to_string(),

            Operation::Unknown { opcode, .. } => format!(".unknown 0x{opcode:02x}"),
        }
    }

    /// Check if this is a control flow instruction
    #[must_use]
    pub fn is_control_flow(&self) -> bool {
        matches!(
            self.operation,
            Operation::If { .. }
                | Operation::Else
                | Operation::Endif
                | Operation::Loop
                | Operation::Break { .. }
                | Operation::Continue { .. }
                | Operation::Endloop
                | Operation::Call { .. }
                | Operation::Return
                | Operation::Halt
        )
    }

    /// Check if this is a memory operation
    #[must_use]
    pub fn is_memory(&self) -> bool {
        matches!(
            self.operation,
            Operation::LocalLoad { .. }
                | Operation::LocalStore { .. }
                | Operation::DeviceLoad { .. }
                | Operation::DeviceStore { .. }
                | Operation::LocalAtomic { .. }
                | Operation::LocalAtomicCas { .. }
                | Operation::DeviceAtomic { .. }
                | Operation::DeviceAtomicCas { .. }
        )
    }

    /// Check if this is a barrier or fence
    #[must_use]
    pub fn is_sync(&self) -> bool {
        matches!(
            self.operation,
            Operation::Barrier
                | Operation::FenceAcquire { .. }
                | Operation::FenceRelease { .. }
                | Operation::FenceAcqRel { .. }
                | Operation::Wait
        )
    }

    /// Check if this is a wave operation
    #[must_use]
    pub fn is_wave_op(&self) -> bool {
        matches!(
            self.operation,
            Operation::WaveOp { .. }
                | Operation::WaveReduce { .. }
                | Operation::WaveBallot { .. }
                | Operation::WaveVote { .. }
        )
    }
}

/// Helper to get raw opcode and modifier from instruction
#[must_use]
pub fn extract_opcode_modifier(instruction: &DecodedInstruction) -> (Opcode, Option<u8>) {
    match &instruction.operation {
        Operation::Iadd { .. } => (Opcode::Iadd, None),
        Operation::Isub { .. } => (Opcode::Isub, None),
        Operation::Imul { .. } => (Opcode::Imul, None),
        Operation::ImulHi { .. } => (Opcode::ImulHi, None),
        Operation::Imad { .. } => (Opcode::Imad, None),
        Operation::Idiv { .. } => (Opcode::Idiv, None),
        Operation::Imod { .. } => (Opcode::Imod, None),
        Operation::Ineg { .. } => (Opcode::Ineg, None),
        Operation::Iabs { .. } => (Opcode::Iabs, None),
        Operation::Imin { .. } => (Opcode::Imin, None),
        Operation::Imax { .. } => (Opcode::Imax, None),
        Operation::Iclamp { .. } => (Opcode::Iclamp, None),

        Operation::Fadd { .. } => (Opcode::Fadd, None),
        Operation::Fsub { .. } => (Opcode::Fsub, None),
        Operation::Fmul { .. } => (Opcode::Fmul, None),
        Operation::Fma { .. } => (Opcode::Fma, None),
        Operation::Fdiv { .. } => (Opcode::Fdiv, None),
        Operation::Fneg { .. } => (Opcode::Fneg, None),
        Operation::Fabs { .. } => (Opcode::Fabs, None),
        Operation::Fmin { .. } => (Opcode::Fmin, None),
        Operation::Fmax { .. } => (Opcode::Fmax, None),
        Operation::Fclamp { .. } => (Opcode::Fclamp, None),
        Operation::Fsqrt { .. } => (Opcode::Fsqrt, None),
        Operation::FUnary { op, .. } => (Opcode::FUnaryOps, Some(*op as u8)),

        Operation::F16 { op, .. } => (Opcode::F16Ops, Some(*op as u8)),
        Operation::F16Packed { op, .. } => (Opcode::F16PackedOps, Some(*op as u8)),
        Operation::Bf16 { op, .. } => (Opcode::Bf16Ops, Some(*op as u8)),
        Operation::Bf16Packed { op, .. } => (Opcode::Bf16PackedOps, Some(*op as u8)),
        Operation::F64 { op, .. } => (Opcode::F64Ops, Some(*op as u8)),
        Operation::F64DivSqrt { op, .. } => (Opcode::F64DivSqrt, Some(*op as u8)),

        Operation::And { .. } => (Opcode::And, None),
        Operation::Or { .. } => (Opcode::Or, None),
        Operation::Xor { .. } => (Opcode::Xor, None),
        Operation::Not { .. } => (Opcode::Not, None),
        Operation::Shl { .. } => (Opcode::Shl, None),
        Operation::Shr { .. } => (Opcode::Shr, None),
        Operation::Sar { .. } => (Opcode::Sar, None),
        Operation::BitOp { op, .. } => (Opcode::BitOps, Some(*op as u8)),

        Operation::Icmp { op, .. } => (Opcode::Icmp, Some(*op as u8)),
        Operation::Ucmp { op, .. } => (Opcode::Ucmp, Some(*op as u8)),
        Operation::Fcmp { op, .. } => (Opcode::Fcmp, Some(*op as u8)),

        Operation::Select { .. } => (Opcode::Select, None),
        Operation::Cvt { cvt_type, .. } => (Opcode::Cvt, Some(*cvt_type as u8)),

        Operation::LocalLoad { width, .. } => (Opcode::LocalLoad, Some(*width as u8)),
        Operation::LocalStore { width, .. } => (Opcode::LocalStore, Some(*width as u8)),
        Operation::DeviceLoad { width, .. } => (Opcode::DeviceLoad, Some(*width as u8)),
        Operation::DeviceStore { width, .. } => (Opcode::DeviceStore, Some(*width as u8)),

        Operation::LocalAtomic { op, .. } => (Opcode::LocalAtomic, Some(*op as u8)),
        Operation::LocalAtomicCas { .. } => (Opcode::LocalAtomic, None),
        Operation::DeviceAtomic { op, .. } => (Opcode::DeviceAtomic, Some(*op as u8)),
        Operation::DeviceAtomicCas { .. } => (Opcode::DeviceAtomic, None),

        Operation::WaveOp { op, .. } | Operation::WaveVote { op, .. } => {
            (Opcode::WaveOp, Some(*op as u8))
        }
        Operation::WaveReduce { op, .. } => (Opcode::WaveOp, Some(*op as u8)),
        Operation::WaveBallot { .. } => (Opcode::WaveOp, Some(WaveOpType::Ballot as u8)),

        Operation::If { .. } => (Opcode::Control, Some(ControlOp::If as u8)),
        Operation::Else => (Opcode::Control, Some(ControlOp::Else as u8)),
        Operation::Endif => (Opcode::Control, Some(ControlOp::Endif as u8)),
        Operation::Loop => (Opcode::Control, Some(ControlOp::Loop as u8)),
        Operation::Break { .. } => (Opcode::Control, Some(ControlOp::Break as u8)),
        Operation::Continue { .. } => (Opcode::Control, Some(ControlOp::Continue as u8)),
        Operation::Endloop => (Opcode::Control, Some(ControlOp::Endloop as u8)),
        Operation::Call { .. } => (Opcode::Control, Some(ControlOp::Call as u8)),

        Operation::Return => (Opcode::Control, Some(SyncOp::Return as u8)),
        Operation::Halt => (Opcode::Control, Some(SyncOp::Halt as u8)),
        Operation::Barrier => (Opcode::Control, Some(SyncOp::Barrier as u8)),
        Operation::FenceAcquire { .. } => (Opcode::Control, Some(SyncOp::FenceAcquire as u8)),
        Operation::FenceRelease { .. } => (Opcode::Control, Some(SyncOp::FenceRelease as u8)),
        Operation::FenceAcqRel { .. } => (Opcode::Control, Some(SyncOp::FenceAcqRel as u8)),
        Operation::Wait => (Opcode::Control, Some(SyncOp::Wait as u8)),
        Operation::Nop => (Opcode::Control, Some(SyncOp::Nop as u8)),

        Operation::Mov { .. } => (Opcode::Control, Some(MiscOp::Mov as u8)),
        Operation::MovImm { .. } => (Opcode::Control, Some(MiscOp::MovImm as u8)),
        Operation::MovSr { .. } => (Opcode::Control, Some(MiscOp::MovSr as u8)),

        Operation::MmaLoadA { .. } => (Opcode::Mma, Some(MmaOp::LoadA as u8)),
        Operation::MmaLoadB { .. } => (Opcode::Mma, Some(MmaOp::LoadB as u8)),
        Operation::MmaStoreC { .. } => (Opcode::Mma, Some(MmaOp::StoreC as u8)),
        Operation::MmaCompute { .. } => (Opcode::Mma, Some(MmaOp::Compute as u8)),

        Operation::Unknown { opcode, .. } => {
            Opcode::from_u8(*opcode).map_or((Opcode::Control, None), |op| (op, None))
        }
    }
}
