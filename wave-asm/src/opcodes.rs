// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Opcode definitions and instruction signature tables. Maps mnemonic strings to instruction
//! encoding metadata, opcode, operand types, modifiers. Uses `LazyLock` for static
//! initialization of the lookup tables.

use std::collections::HashMap;
use std::sync::LazyLock;

pub const OPCODE_SHIFT: u32 = 24;
pub const OPCODE_MASK: u32 = 0xFF;
pub const RD_SHIFT: u32 = 16;
pub const RD_MASK: u32 = 0xFF;
pub const RS1_SHIFT: u32 = 8;
pub const RS1_MASK: u32 = 0xFF;
pub const MODIFIER_SHIFT: u32 = 4;
pub const MODIFIER_MASK: u32 = 0x0F;
pub const PRED_REG_SHIFT: u32 = 0;
pub const PRED_REG_MASK: u32 = 0x03;
pub const PRED_NEG_SHIFT: u32 = 2;
pub const PRED_NEG_MASK: u32 = 0x01;
pub const EXTENDED_SCOPE_SHIFT: u32 = 0;
pub const EXTENDED_SCOPE_MASK: u32 = 0x03;

pub const EXTENDED_RS2_SHIFT: u32 = 24;
pub const EXTENDED_RS2_MASK: u32 = 0xFF;
pub const EXTENDED_RS3_SHIFT: u32 = 16;
pub const EXTENDED_RS3_MASK: u32 = 0xFF;
pub const EXTENDED_RS4_SHIFT: u32 = 8;
pub const EXTENDED_RS4_MASK: u32 = 0xFF;

pub const MAX_REGISTER_INDEX: u8 = 254;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Opcode {
    Iadd = 0x00,
    Isub = 0x01,
    Imul = 0x02,
    ImulHi = 0x03,
    Imad = 0x04,
    Idiv = 0x05,
    Imod = 0x06,
    Ineg = 0x07,
    Iabs = 0x08,
    Imin = 0x09,
    Imax = 0x0A,
    Iclamp = 0x0B,

    Fadd = 0x10,
    Fsub = 0x11,
    Fmul = 0x12,
    Fma = 0x13,
    Fdiv = 0x14,
    Fneg = 0x15,
    Fabs = 0x16,
    Fmin = 0x17,
    Fmax = 0x18,
    Fclamp = 0x19,
    Fsqrt = 0x1A,
    FUnaryOps = 0x1B,

    F16Ops = 0x1C,
    F16PackedOps = 0x1D,

    F64Ops = 0x1E,
    F64DivSqrt = 0x1F,

    Bf16Ops = 0x2D,
    Bf16PackedOps = 0x2E,

    And = 0x20,
    Or = 0x21,
    Xor = 0x22,
    Not = 0x23,
    Shl = 0x24,
    Shr = 0x25,
    Sar = 0x26,
    BitOps = 0x27,

    Icmp = 0x28,
    Ucmp = 0x29,
    Fcmp = 0x2A,
    Select = 0x2B,
    Cvt = 0x2C,

    LocalLoad = 0x30,
    LocalStore = 0x31,

    DeviceLoad = 0x38,
    DeviceStore = 0x39,

    LocalAtomic = 0x3C,
    DeviceAtomic = 0x3D,

    WaveOp = 0x3E,

    Control = 0x3F,

    Mma = 0x40,

    Misc = 0x41,
}

impl Opcode {
    #[must_use]
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FUnaryOp {
    Frsqrt = 0,
    Frcp = 1,
    Ffloor = 2,
    Fceil = 3,
    Fround = 4,
    Ftrunc = 5,
    Ffract = 6,
    Fsat = 7,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F16Op {
    Hadd = 0,
    Hsub = 1,
    Hmul = 2,
    Hma = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F16PackedOp {
    Hadd2 = 0,
    Hmul2 = 1,
    Hma2 = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bf16Op {
    Badd = 0,
    Bsub = 1,
    Bmul = 2,
    Bma = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bf16PackedOp {
    Badd2 = 0,
    Bmul2 = 1,
    Bma2 = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F64Op {
    Dadd = 0,
    Dsub = 1,
    Dmul = 2,
    Dma = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F64DivSqrtOp {
    Ddiv = 0,
    Dsqrt = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitOpType {
    Bitcount = 0,
    Bitfind = 1,
    Bitrev = 2,
    Bfe = 3,
    Bfi = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq = 0,
    Ne = 1,
    Lt = 2,
    Le = 3,
    Gt = 4,
    Ge = 5,
    Ord = 6,
    Unord = 7,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CvtType {
    F32I32 = 0,
    F32U32 = 1,
    I32F32 = 2,
    U32F32 = 3,
    F32F16 = 4,
    F16F32 = 5,
    F32F64 = 6,
    F64F32 = 7,
    F32Bf16 = 8,
    Bf16F32 = 9,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemWidth {
    U8 = 0,
    U16 = 1,
    U32 = 2,
    U64 = 3,
    U128 = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheHint {
    Cached = 0,
    Uncached = 1,
    Streaming = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicOp {
    Add = 0,
    Sub = 1,
    Min = 2,
    Max = 3,
    And = 4,
    Or = 5,
    Xor = 6,
    Exchange = 7,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    Wave = 0,
    Workgroup = 1,
    Device = 2,
    System = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveOpType {
    Shuffle = 0,
    ShuffleUp = 1,
    ShuffleDown = 2,
    ShuffleXor = 3,
    Broadcast = 4,
    Ballot = 5,
    Any = 6,
    All = 7,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveReduceType {
    PrefixSum = 0,
    ReduceAdd = 1,
    ReduceMin = 2,
    ReduceMax = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlOp {
    If = 0,
    Else = 1,
    Endif = 2,
    Loop = 3,
    Break = 4,
    Continue = 5,
    Endloop = 6,
    Call = 7,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncOp {
    Return = 0,
    Halt = 1,
    Barrier = 2,
    FenceAcquire = 3,
    FenceRelease = 4,
    FenceAcqRel = 5,
    Wait = 6,
    Nop = 7,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiscOp {
    Mov = 0,
    MovImm = 1,
    MovSr = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandKind {
    Rd,
    Rs1,
    Rs2,
    Rs3,
    Rs4,
    Pd,
    PdSrc,
    Imm32,
    Label,
    Scope,
    OptionalRd,
    SpecialReg,
}

#[derive(Debug, Clone)]
pub struct InstructionSignature {
    pub opcode: Opcode,
    pub operands: &'static [OperandKind],
    pub modifier: Option<u8>,
    pub extended: bool,
    pub wave_reduce: bool,
}

pub static MNEMONIC_MAP: LazyLock<HashMap<&'static str, InstructionSignature>> =
    LazyLock::new(build_mnemonic_map);

const INSTRUCTION_SIGNATURES: &[(&str, InstructionSignature)] = &[
    (
        "iadd",
        InstructionSignature {
            opcode: Opcode::Iadd,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "isub",
        InstructionSignature {
            opcode: Opcode::Isub,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "imul",
        InstructionSignature {
            opcode: Opcode::Imul,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "imul_hi",
        InstructionSignature {
            opcode: Opcode::ImulHi,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "imad",
        InstructionSignature {
            opcode: Opcode::Imad,
            operands: &[
                OperandKind::Rd,
                OperandKind::Rs1,
                OperandKind::Rs2,
                OperandKind::Rs3,
            ],
            modifier: None,
            extended: true,
            wave_reduce: false,
        },
    ),
    (
        "idiv",
        InstructionSignature {
            opcode: Opcode::Idiv,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "imod",
        InstructionSignature {
            opcode: Opcode::Imod,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "ineg",
        InstructionSignature {
            opcode: Opcode::Ineg,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "iabs",
        InstructionSignature {
            opcode: Opcode::Iabs,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "imin",
        InstructionSignature {
            opcode: Opcode::Imin,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "imax",
        InstructionSignature {
            opcode: Opcode::Imax,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "iclamp",
        InstructionSignature {
            opcode: Opcode::Iclamp,
            operands: &[
                OperandKind::Rd,
                OperandKind::Rs1,
                OperandKind::Rs2,
                OperandKind::Rs3,
            ],
            modifier: None,
            extended: true,
            wave_reduce: false,
        },
    ),
    (
        "fadd",
        InstructionSignature {
            opcode: Opcode::Fadd,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "fsub",
        InstructionSignature {
            opcode: Opcode::Fsub,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "fmul",
        InstructionSignature {
            opcode: Opcode::Fmul,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "fma",
        InstructionSignature {
            opcode: Opcode::Fma,
            operands: &[
                OperandKind::Rd,
                OperandKind::Rs1,
                OperandKind::Rs2,
                OperandKind::Rs3,
            ],
            modifier: None,
            extended: true,
            wave_reduce: false,
        },
    ),
    (
        "fdiv",
        InstructionSignature {
            opcode: Opcode::Fdiv,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "fneg",
        InstructionSignature {
            opcode: Opcode::Fneg,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "fabs",
        InstructionSignature {
            opcode: Opcode::Fabs,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "fmin",
        InstructionSignature {
            opcode: Opcode::Fmin,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "fmax",
        InstructionSignature {
            opcode: Opcode::Fmax,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "fclamp",
        InstructionSignature {
            opcode: Opcode::Fclamp,
            operands: &[
                OperandKind::Rd,
                OperandKind::Rs1,
                OperandKind::Rs2,
                OperandKind::Rs3,
            ],
            modifier: None,
            extended: true,
            wave_reduce: false,
        },
    ),
    (
        "fsqrt",
        InstructionSignature {
            opcode: Opcode::Fsqrt,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "frsqrt",
        InstructionSignature {
            opcode: Opcode::FUnaryOps,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(FUnaryOp::Frsqrt as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "frcp",
        InstructionSignature {
            opcode: Opcode::FUnaryOps,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(FUnaryOp::Frcp as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "ffloor",
        InstructionSignature {
            opcode: Opcode::FUnaryOps,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(FUnaryOp::Ffloor as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "fceil",
        InstructionSignature {
            opcode: Opcode::FUnaryOps,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(FUnaryOp::Fceil as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "fround",
        InstructionSignature {
            opcode: Opcode::FUnaryOps,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(FUnaryOp::Fround as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "ftrunc",
        InstructionSignature {
            opcode: Opcode::FUnaryOps,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(FUnaryOp::Ftrunc as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "ffract",
        InstructionSignature {
            opcode: Opcode::FUnaryOps,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(FUnaryOp::Ffract as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "fsat",
        InstructionSignature {
            opcode: Opcode::FUnaryOps,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(FUnaryOp::Fsat as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "fsin",
        InstructionSignature {
            opcode: Opcode::FUnaryOps,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "fcos",
        InstructionSignature {
            opcode: Opcode::FUnaryOps,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(9),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "fexp2",
        InstructionSignature {
            opcode: Opcode::FUnaryOps,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(10),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "flog2",
        InstructionSignature {
            opcode: Opcode::FUnaryOps,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(11),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "hadd",
        InstructionSignature {
            opcode: Opcode::F16Ops,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(F16Op::Hadd as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "hsub",
        InstructionSignature {
            opcode: Opcode::F16Ops,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(F16Op::Hsub as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "hmul",
        InstructionSignature {
            opcode: Opcode::F16Ops,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(F16Op::Hmul as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "hma",
        InstructionSignature {
            opcode: Opcode::F16Ops,
            operands: &[
                OperandKind::Rd,
                OperandKind::Rs1,
                OperandKind::Rs2,
                OperandKind::Rs3,
            ],
            modifier: Some(F16Op::Hma as u8),
            extended: true,
            wave_reduce: false,
        },
    ),
    (
        "hadd2",
        InstructionSignature {
            opcode: Opcode::F16PackedOps,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(F16PackedOp::Hadd2 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "hmul2",
        InstructionSignature {
            opcode: Opcode::F16PackedOps,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(F16PackedOp::Hmul2 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "hma2",
        InstructionSignature {
            opcode: Opcode::F16PackedOps,
            operands: &[
                OperandKind::Rd,
                OperandKind::Rs1,
                OperandKind::Rs2,
                OperandKind::Rs3,
            ],
            modifier: Some(F16PackedOp::Hma2 as u8),
            extended: true,
            wave_reduce: false,
        },
    ),
    (
        "badd",
        InstructionSignature {
            opcode: Opcode::Bf16Ops,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(Bf16Op::Badd as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "bsub",
        InstructionSignature {
            opcode: Opcode::Bf16Ops,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(Bf16Op::Bsub as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "bmul",
        InstructionSignature {
            opcode: Opcode::Bf16Ops,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(Bf16Op::Bmul as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "bma",
        InstructionSignature {
            opcode: Opcode::Bf16Ops,
            operands: &[
                OperandKind::Rd,
                OperandKind::Rs1,
                OperandKind::Rs2,
                OperandKind::Rs3,
            ],
            modifier: Some(Bf16Op::Bma as u8),
            extended: true,
            wave_reduce: false,
        },
    ),
    (
        "badd2",
        InstructionSignature {
            opcode: Opcode::Bf16PackedOps,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(Bf16PackedOp::Badd2 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "bmul2",
        InstructionSignature {
            opcode: Opcode::Bf16PackedOps,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(Bf16PackedOp::Bmul2 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "bma2",
        InstructionSignature {
            opcode: Opcode::Bf16PackedOps,
            operands: &[
                OperandKind::Rd,
                OperandKind::Rs1,
                OperandKind::Rs2,
                OperandKind::Rs3,
            ],
            modifier: Some(Bf16PackedOp::Bma2 as u8),
            extended: true,
            wave_reduce: false,
        },
    ),
    (
        "dadd",
        InstructionSignature {
            opcode: Opcode::F64Ops,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(F64Op::Dadd as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "dsub",
        InstructionSignature {
            opcode: Opcode::F64Ops,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(F64Op::Dsub as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "dmul",
        InstructionSignature {
            opcode: Opcode::F64Ops,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(F64Op::Dmul as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "dma",
        InstructionSignature {
            opcode: Opcode::F64Ops,
            operands: &[
                OperandKind::Rd,
                OperandKind::Rs1,
                OperandKind::Rs2,
                OperandKind::Rs3,
            ],
            modifier: Some(F64Op::Dma as u8),
            extended: true,
            wave_reduce: false,
        },
    ),
    (
        "ddiv",
        InstructionSignature {
            opcode: Opcode::F64DivSqrt,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(F64DivSqrtOp::Ddiv as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "dsqrt",
        InstructionSignature {
            opcode: Opcode::F64DivSqrt,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(F64DivSqrtOp::Dsqrt as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "and",
        InstructionSignature {
            opcode: Opcode::And,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "or",
        InstructionSignature {
            opcode: Opcode::Or,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "xor",
        InstructionSignature {
            opcode: Opcode::Xor,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "not",
        InstructionSignature {
            opcode: Opcode::Not,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "shl",
        InstructionSignature {
            opcode: Opcode::Shl,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "shr",
        InstructionSignature {
            opcode: Opcode::Shr,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "sar",
        InstructionSignature {
            opcode: Opcode::Sar,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "bitcount",
        InstructionSignature {
            opcode: Opcode::BitOps,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(BitOpType::Bitcount as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "bitfind",
        InstructionSignature {
            opcode: Opcode::BitOps,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(BitOpType::Bitfind as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "bitrev",
        InstructionSignature {
            opcode: Opcode::BitOps,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(BitOpType::Bitrev as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "bfe",
        InstructionSignature {
            opcode: Opcode::BitOps,
            operands: &[
                OperandKind::Rd,
                OperandKind::Rs1,
                OperandKind::Rs2,
                OperandKind::Rs3,
            ],
            modifier: Some(BitOpType::Bfe as u8),
            extended: true,
            wave_reduce: false,
        },
    ),
    (
        "bfi",
        InstructionSignature {
            opcode: Opcode::BitOps,
            operands: &[
                OperandKind::Rd,
                OperandKind::Rs1,
                OperandKind::Rs2,
                OperandKind::Rs3,
                OperandKind::Rs4,
            ],
            modifier: Some(BitOpType::Bfi as u8),
            extended: true,

            wave_reduce: false,
        },
    ),
    (
        "icmp_eq",
        InstructionSignature {
            opcode: Opcode::Icmp,
            operands: &[OperandKind::Pd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(CmpOp::Eq as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "icmp_ne",
        InstructionSignature {
            opcode: Opcode::Icmp,
            operands: &[OperandKind::Pd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(CmpOp::Ne as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "icmp_lt",
        InstructionSignature {
            opcode: Opcode::Icmp,
            operands: &[OperandKind::Pd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(CmpOp::Lt as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "icmp_le",
        InstructionSignature {
            opcode: Opcode::Icmp,
            operands: &[OperandKind::Pd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(CmpOp::Le as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "icmp_gt",
        InstructionSignature {
            opcode: Opcode::Icmp,
            operands: &[OperandKind::Pd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(CmpOp::Gt as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "icmp_ge",
        InstructionSignature {
            opcode: Opcode::Icmp,
            operands: &[OperandKind::Pd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(CmpOp::Ge as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "ucmp_lt",
        InstructionSignature {
            opcode: Opcode::Ucmp,
            operands: &[OperandKind::Pd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(CmpOp::Lt as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "ucmp_le",
        InstructionSignature {
            opcode: Opcode::Ucmp,
            operands: &[OperandKind::Pd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(CmpOp::Le as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "fcmp_eq",
        InstructionSignature {
            opcode: Opcode::Fcmp,
            operands: &[OperandKind::Pd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(CmpOp::Eq as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "fcmp_ne",
        InstructionSignature {
            opcode: Opcode::Fcmp,
            operands: &[OperandKind::Pd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(CmpOp::Ne as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "fcmp_lt",
        InstructionSignature {
            opcode: Opcode::Fcmp,
            operands: &[OperandKind::Pd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(CmpOp::Lt as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "fcmp_le",
        InstructionSignature {
            opcode: Opcode::Fcmp,
            operands: &[OperandKind::Pd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(CmpOp::Le as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "fcmp_gt",
        InstructionSignature {
            opcode: Opcode::Fcmp,
            operands: &[OperandKind::Pd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(CmpOp::Gt as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "fcmp_ord",
        InstructionSignature {
            opcode: Opcode::Fcmp,
            operands: &[OperandKind::Pd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(CmpOp::Ord as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "fcmp_unord",
        InstructionSignature {
            opcode: Opcode::Fcmp,
            operands: &[OperandKind::Pd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(CmpOp::Unord as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "select",
        InstructionSignature {
            opcode: Opcode::Select,
            operands: &[
                OperandKind::Rd,
                OperandKind::PdSrc,
                OperandKind::Rs1,
                OperandKind::Rs2,
            ],
            modifier: None,
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "cvt_f32_i32",
        InstructionSignature {
            opcode: Opcode::Cvt,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(CvtType::F32I32 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "cvt_f32_u32",
        InstructionSignature {
            opcode: Opcode::Cvt,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(CvtType::F32U32 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "cvt_i32_f32",
        InstructionSignature {
            opcode: Opcode::Cvt,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(CvtType::I32F32 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "cvt_u32_f32",
        InstructionSignature {
            opcode: Opcode::Cvt,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(CvtType::U32F32 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "cvt_f32_f16",
        InstructionSignature {
            opcode: Opcode::Cvt,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(CvtType::F32F16 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "cvt_f16_f32",
        InstructionSignature {
            opcode: Opcode::Cvt,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(CvtType::F16F32 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "cvt_f32_f64",
        InstructionSignature {
            opcode: Opcode::Cvt,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(CvtType::F32F64 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "cvt_f64_f32",
        InstructionSignature {
            opcode: Opcode::Cvt,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(CvtType::F64F32 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "cvt_f32_bf16",
        InstructionSignature {
            opcode: Opcode::Cvt,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(CvtType::F32Bf16 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "cvt_bf16_f32",
        InstructionSignature {
            opcode: Opcode::Cvt,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(CvtType::Bf16F32 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "local_load_u8",
        InstructionSignature {
            opcode: Opcode::LocalLoad,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(MemWidth::U8 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "local_load_u16",
        InstructionSignature {
            opcode: Opcode::LocalLoad,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(MemWidth::U16 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "local_load_u32",
        InstructionSignature {
            opcode: Opcode::LocalLoad,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(MemWidth::U32 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "local_load_u64",
        InstructionSignature {
            opcode: Opcode::LocalLoad,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(MemWidth::U64 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "local_store_u8",
        InstructionSignature {
            opcode: Opcode::LocalStore,
            operands: &[OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(MemWidth::U8 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "local_store_u16",
        InstructionSignature {
            opcode: Opcode::LocalStore,
            operands: &[OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(MemWidth::U16 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "local_store_u32",
        InstructionSignature {
            opcode: Opcode::LocalStore,
            operands: &[OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(MemWidth::U32 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "local_store_u64",
        InstructionSignature {
            opcode: Opcode::LocalStore,
            operands: &[OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(MemWidth::U64 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "device_load_u8",
        InstructionSignature {
            opcode: Opcode::DeviceLoad,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(MemWidth::U8 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "device_load_u16",
        InstructionSignature {
            opcode: Opcode::DeviceLoad,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(MemWidth::U16 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "device_load_u32",
        InstructionSignature {
            opcode: Opcode::DeviceLoad,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(MemWidth::U32 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "device_load_u64",
        InstructionSignature {
            opcode: Opcode::DeviceLoad,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(MemWidth::U64 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "device_load_u128",
        InstructionSignature {
            opcode: Opcode::DeviceLoad,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(MemWidth::U128 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "device_store_u8",
        InstructionSignature {
            opcode: Opcode::DeviceStore,
            operands: &[OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(MemWidth::U8 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "device_store_u16",
        InstructionSignature {
            opcode: Opcode::DeviceStore,
            operands: &[OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(MemWidth::U16 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "device_store_u32",
        InstructionSignature {
            opcode: Opcode::DeviceStore,
            operands: &[OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(MemWidth::U32 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "device_store_u64",
        InstructionSignature {
            opcode: Opcode::DeviceStore,
            operands: &[OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(MemWidth::U64 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "device_store_u128",
        InstructionSignature {
            opcode: Opcode::DeviceStore,
            operands: &[OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(MemWidth::U128 as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "local_atomic_add",
        InstructionSignature {
            opcode: Opcode::LocalAtomic,
            operands: &[OperandKind::OptionalRd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(AtomicOp::Add as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "local_atomic_sub",
        InstructionSignature {
            opcode: Opcode::LocalAtomic,
            operands: &[OperandKind::OptionalRd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(AtomicOp::Sub as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "local_atomic_min",
        InstructionSignature {
            opcode: Opcode::LocalAtomic,
            operands: &[OperandKind::OptionalRd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(AtomicOp::Min as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "local_atomic_max",
        InstructionSignature {
            opcode: Opcode::LocalAtomic,
            operands: &[OperandKind::OptionalRd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(AtomicOp::Max as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "local_atomic_and",
        InstructionSignature {
            opcode: Opcode::LocalAtomic,
            operands: &[OperandKind::OptionalRd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(AtomicOp::And as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "local_atomic_or",
        InstructionSignature {
            opcode: Opcode::LocalAtomic,
            operands: &[OperandKind::OptionalRd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(AtomicOp::Or as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "local_atomic_xor",
        InstructionSignature {
            opcode: Opcode::LocalAtomic,
            operands: &[OperandKind::OptionalRd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(AtomicOp::Xor as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "local_atomic_exchange",
        InstructionSignature {
            opcode: Opcode::LocalAtomic,
            operands: &[OperandKind::OptionalRd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(AtomicOp::Exchange as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "local_atomic_cas",
        InstructionSignature {
            opcode: Opcode::LocalAtomic,
            operands: &[
                OperandKind::OptionalRd,
                OperandKind::Rs1,
                OperandKind::Rs2,
                OperandKind::Rs3,
            ],
            modifier: Some(8),
            extended: true,

            wave_reduce: false,
        },
    ),
    (
        "atomic_add",
        InstructionSignature {
            opcode: Opcode::DeviceAtomic,
            operands: &[
                OperandKind::OptionalRd,
                OperandKind::Rs1,
                OperandKind::Rs2,
                OperandKind::Scope,
            ],
            modifier: Some(AtomicOp::Add as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "atomic_sub",
        InstructionSignature {
            opcode: Opcode::DeviceAtomic,
            operands: &[
                OperandKind::OptionalRd,
                OperandKind::Rs1,
                OperandKind::Rs2,
                OperandKind::Scope,
            ],
            modifier: Some(AtomicOp::Sub as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "atomic_min",
        InstructionSignature {
            opcode: Opcode::DeviceAtomic,
            operands: &[
                OperandKind::OptionalRd,
                OperandKind::Rs1,
                OperandKind::Rs2,
                OperandKind::Scope,
            ],
            modifier: Some(AtomicOp::Min as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "atomic_max",
        InstructionSignature {
            opcode: Opcode::DeviceAtomic,
            operands: &[
                OperandKind::OptionalRd,
                OperandKind::Rs1,
                OperandKind::Rs2,
                OperandKind::Scope,
            ],
            modifier: Some(AtomicOp::Max as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "atomic_and",
        InstructionSignature {
            opcode: Opcode::DeviceAtomic,
            operands: &[
                OperandKind::OptionalRd,
                OperandKind::Rs1,
                OperandKind::Rs2,
                OperandKind::Scope,
            ],
            modifier: Some(AtomicOp::And as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "atomic_or",
        InstructionSignature {
            opcode: Opcode::DeviceAtomic,
            operands: &[
                OperandKind::OptionalRd,
                OperandKind::Rs1,
                OperandKind::Rs2,
                OperandKind::Scope,
            ],
            modifier: Some(AtomicOp::Or as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "atomic_xor",
        InstructionSignature {
            opcode: Opcode::DeviceAtomic,
            operands: &[
                OperandKind::OptionalRd,
                OperandKind::Rs1,
                OperandKind::Rs2,
                OperandKind::Scope,
            ],
            modifier: Some(AtomicOp::Xor as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "atomic_exchange",
        InstructionSignature {
            opcode: Opcode::DeviceAtomic,
            operands: &[
                OperandKind::OptionalRd,
                OperandKind::Rs1,
                OperandKind::Rs2,
                OperandKind::Scope,
            ],
            modifier: Some(AtomicOp::Exchange as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "atomic_cas",
        InstructionSignature {
            opcode: Opcode::DeviceAtomic,
            operands: &[
                OperandKind::OptionalRd,
                OperandKind::Rs1,
                OperandKind::Rs2,
                OperandKind::Rs3,
                OperandKind::Scope,
            ],
            modifier: Some(8),
            extended: true,

            wave_reduce: false,
        },
    ),
    (
        "wave_shuffle",
        InstructionSignature {
            opcode: Opcode::WaveOp,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(WaveOpType::Shuffle as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "wave_shuffle_up",
        InstructionSignature {
            opcode: Opcode::WaveOp,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(WaveOpType::ShuffleUp as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "wave_shuffle_down",
        InstructionSignature {
            opcode: Opcode::WaveOp,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(WaveOpType::ShuffleDown as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "wave_shuffle_xor",
        InstructionSignature {
            opcode: Opcode::WaveOp,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(WaveOpType::ShuffleXor as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "wave_broadcast",
        InstructionSignature {
            opcode: Opcode::WaveOp,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(WaveOpType::Broadcast as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "wave_ballot",
        InstructionSignature {
            opcode: Opcode::WaveOp,
            operands: &[OperandKind::Rd, OperandKind::PdSrc],
            modifier: Some(WaveOpType::Ballot as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "wave_any",
        InstructionSignature {
            opcode: Opcode::WaveOp,
            operands: &[OperandKind::Pd, OperandKind::PdSrc],
            modifier: Some(WaveOpType::Any as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "wave_all",
        InstructionSignature {
            opcode: Opcode::WaveOp,
            operands: &[OperandKind::Pd, OperandKind::PdSrc],
            modifier: Some(WaveOpType::All as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "wave_prefix_sum",
        InstructionSignature {
            opcode: Opcode::WaveOp,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(WaveReduceType::PrefixSum as u8 + 8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "wave_reduce_add",
        InstructionSignature {
            opcode: Opcode::WaveOp,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(WaveReduceType::ReduceAdd as u8 + 8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "wave_reduce_min",
        InstructionSignature {
            opcode: Opcode::WaveOp,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(WaveReduceType::ReduceMin as u8 + 8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "wave_reduce_max",
        InstructionSignature {
            opcode: Opcode::WaveOp,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(WaveReduceType::ReduceMax as u8 + 8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "mma_load_a",
        InstructionSignature {
            opcode: Opcode::Mma,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(0),
            extended: true,
            wave_reduce: false,
        },
    ),
    (
        "mma_load_b",
        InstructionSignature {
            opcode: Opcode::Mma,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(1),
            extended: true,
            wave_reduce: false,
        },
    ),
    (
        "mma_store_c",
        InstructionSignature {
            opcode: Opcode::Mma,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(2),
            extended: true,
            wave_reduce: false,
        },
    ),
    (
        "mma_compute",
        InstructionSignature {
            opcode: Opcode::Mma,
            operands: &[OperandKind::Rd, OperandKind::Rs1, OperandKind::Rs2],
            modifier: Some(3),
            extended: true,
            wave_reduce: false,
        },
    ),
    (
        "if",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[OperandKind::PdSrc],
            modifier: Some(ControlOp::If as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "else",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[],
            modifier: Some(ControlOp::Else as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "endif",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[],
            modifier: Some(ControlOp::Endif as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "loop",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[],
            modifier: Some(ControlOp::Loop as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "break",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[OperandKind::PdSrc],
            modifier: Some(ControlOp::Break as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "continue",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[OperandKind::PdSrc],
            modifier: Some(ControlOp::Continue as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "endloop",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[],
            modifier: Some(ControlOp::Endloop as u8),
            extended: false,

            wave_reduce: false,
        },
    ),
    (
        "call",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[OperandKind::Label],
            modifier: Some(ControlOp::Call as u8),
            extended: true,

            wave_reduce: false,
        },
    ),
    (
        "return",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[],
            modifier: Some(SyncOp::Return as u8 + 8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "halt",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[],
            modifier: Some(SyncOp::Halt as u8 + 8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "barrier",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[],
            modifier: Some(SyncOp::Barrier as u8 + 8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "fence_acquire",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[OperandKind::Scope],
            modifier: Some(SyncOp::FenceAcquire as u8 + 8),
            extended: true,
            wave_reduce: false,
        },
    ),
    (
        "fence_release",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[OperandKind::Scope],
            modifier: Some(SyncOp::FenceRelease as u8 + 8),
            extended: true,
            wave_reduce: false,
        },
    ),
    (
        "fence_acq_rel",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[OperandKind::Scope],
            modifier: Some(SyncOp::FenceAcqRel as u8 + 8),
            extended: true,
            wave_reduce: false,
        },
    ),
    (
        "wait",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[],
            modifier: Some(SyncOp::Wait as u8 + 8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "nop",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[],
            modifier: Some(SyncOp::Nop as u8 + 8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "mov",
        InstructionSignature {
            opcode: Opcode::Misc,
            operands: &[OperandKind::Rd, OperandKind::Rs1],
            modifier: Some(MiscOp::Mov as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
    (
        "mov_imm",
        InstructionSignature {
            opcode: Opcode::Misc,
            operands: &[OperandKind::Rd, OperandKind::Imm32],
            modifier: Some(MiscOp::MovImm as u8),
            extended: true,
            wave_reduce: false,
        },
    ),
    (
        "mov_sr",
        InstructionSignature {
            opcode: Opcode::Misc,
            operands: &[OperandKind::Rd, OperandKind::SpecialReg],
            modifier: Some(MiscOp::MovSr as u8),
            extended: false,
            wave_reduce: false,
        },
    ),
];

fn build_mnemonic_map() -> HashMap<&'static str, InstructionSignature> {
    INSTRUCTION_SIGNATURES.iter().cloned().collect()
}

pub static SPECIAL_REGISTERS: LazyLock<HashMap<&'static str, u8>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("sr_thread_id_x", 0);
    m.insert("sr_thread_id_y", 1);
    m.insert("sr_thread_id_z", 2);
    m.insert("sr_wave_id", 3);
    m.insert("sr_lane_id", 4);
    m.insert("sr_workgroup_id_x", 5);
    m.insert("sr_workgroup_id_y", 6);
    m.insert("sr_workgroup_id_z", 7);
    m.insert("sr_workgroup_size_x", 8);
    m.insert("sr_workgroup_size_y", 9);
    m.insert("sr_workgroup_size_z", 10);
    m.insert("sr_grid_size_x", 11);
    m.insert("sr_grid_size_y", 12);
    m.insert("sr_grid_size_z", 13);
    m.insert("sr_wave_width", 14);
    m.insert("sr_num_waves", 15);
    m.insert("sr_mma_supported", 16);
    m.insert("sr_mma_m", 17);
    m.insert("sr_mma_n", 18);
    m.insert("sr_mma_k", 19);
    m
});

#[must_use]
pub fn lookup_mnemonic(name: &str) -> Option<&'static InstructionSignature> {
    MNEMONIC_MAP.get(name)
}

#[must_use]
pub fn lookup_special_register(name: &str) -> Option<u8> {
    SPECIAL_REGISTERS.get(name).copied()
}

#[must_use]
pub fn encode_base(
    opcode: u8,
    rd: u8,
    rs1: u8,
    modifier: u8,
    pred_reg: u8,
    negate_predicate: bool,
) -> u32 {
    ((u32::from(opcode) & OPCODE_MASK) << OPCODE_SHIFT)
        | ((u32::from(rd) & RD_MASK) << RD_SHIFT)
        | ((u32::from(rs1) & RS1_MASK) << RS1_SHIFT)
        | ((u32::from(modifier) & MODIFIER_MASK) << MODIFIER_SHIFT)
        | ((u32::from(pred_reg) & PRED_REG_MASK) << PRED_REG_SHIFT)
        | (u32::from(negate_predicate) << PRED_NEG_SHIFT)
}
