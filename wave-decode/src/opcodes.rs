// Copyright (c) 2026 Ojima Abraham. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.
// Opcode definitions and constants for WAVE ISA decoding.
// Mirrors the exact opcode assignments from wave-asm to ensure
// consistency between assembler, disassembler, and emulator.
pub const OPCODE_SHIFT: u32 = 26;
pub const OPCODE_MASK: u32 = 0x3F;
pub const RD_SHIFT: u32 = 21;
pub const RD_MASK: u32 = 0x1F;
pub const RS1_SHIFT: u32 = 16;
pub const RS1_MASK: u32 = 0x1F;
pub const RS2_SHIFT: u32 = 11;
pub const RS2_MASK: u32 = 0x1F;
pub const MODIFIER_SHIFT: u32 = 8;
pub const MODIFIER_MASK: u32 = 0x07;
pub const SCOPE_SHIFT: u32 = 6;
pub const SCOPE_MASK: u32 = 0x03;
pub const PRED_SHIFT: u32 = 4;
pub const PRED_MASK: u32 = 0x03;
pub const PRED_NEG_SHIFT: u32 = 3;
pub const PRED_NEG_MASK: u32 = 0x01;
pub const FLAGS_SHIFT: u32 = 0;
pub const FLAGS_MASK: u32 = 0x07;
pub const EXTENDED_RS3_SHIFT: u32 = 27;
pub const EXTENDED_RS3_MASK: u32 = 0x1F;
pub const EXTENDED_RS4_SHIFT: u32 = 22;
pub const EXTENDED_RS4_MASK: u32 = 0x1F;
pub const SYNC_OP_FLAG: u8 = 0x01;
pub const MISC_OP_FLAG: u8 = 0x02;
pub const WAVE_REDUCE_FLAG: u8 = 0x04;
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
}
impl Opcode {
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x00 => Some(Self::Iadd),
            0x01 => Some(Self::Isub),
            0x02 => Some(Self::Imul),
            0x03 => Some(Self::ImulHi),
            0x04 => Some(Self::Imad),
            0x05 => Some(Self::Idiv),
            0x06 => Some(Self::Imod),
            0x07 => Some(Self::Ineg),
            0x08 => Some(Self::Iabs),
            0x09 => Some(Self::Imin),
            0x0A => Some(Self::Imax),
            0x0B => Some(Self::Iclamp),
            0x10 => Some(Self::Fadd),
            0x11 => Some(Self::Fsub),
            0x12 => Some(Self::Fmul),
            0x13 => Some(Self::Fma),
            0x14 => Some(Self::Fdiv),
            0x15 => Some(Self::Fneg),
            0x16 => Some(Self::Fabs),
            0x17 => Some(Self::Fmin),
            0x18 => Some(Self::Fmax),
            0x19 => Some(Self::Fclamp),
            0x1A => Some(Self::Fsqrt),
            0x1B => Some(Self::FUnaryOps),
            0x1C => Some(Self::F16Ops),
            0x1D => Some(Self::F16PackedOps),
            0x1E => Some(Self::F64Ops),
            0x1F => Some(Self::F64DivSqrt),
            0x20 => Some(Self::And),
            0x21 => Some(Self::Or),
            0x22 => Some(Self::Xor),
            0x23 => Some(Self::Not),
            0x24 => Some(Self::Shl),
            0x25 => Some(Self::Shr),
            0x26 => Some(Self::Sar),
            0x27 => Some(Self::BitOps),
            0x28 => Some(Self::Icmp),
            0x29 => Some(Self::Ucmp),
            0x2A => Some(Self::Fcmp),
            0x2B => Some(Self::Select),
            0x2C => Some(Self::Cvt),
            0x30 => Some(Self::LocalLoad),
            0x31 => Some(Self::LocalStore),
            0x38 => Some(Self::DeviceLoad),
            0x39 => Some(Self::DeviceStore),
            0x3C => Some(Self::LocalAtomic),
            0x3D => Some(Self::DeviceAtomic),
            0x3E => Some(Self::WaveOp),
            0x3F => Some(Self::Control),
            _ => None,
        }
    }
    #[must_use]
    pub fn is_extended(self) -> bool {
        matches!(
            self,
            Self::Imad
                | Self::Iclamp
                | Self::Fma
                | Self::Fclamp
                | Self::F16Ops
                | Self::F16PackedOps
                | Self::F64Ops
                | Self::BitOps
                | Self::LocalAtomic
                | Self::DeviceAtomic
                | Self::Control
        )
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FUnaryOp {
    Frsqrt = 0,
    Frcp = 1,
    Ffloor = 2,
    Fceil = 3,
    Fround = 4,
    Ftrunc = 5,
    Ffract = 6,
    Fsat = 7,
    Fsin = 8,
    Fcos = 9,
    Fexp2 = 10,
    Flog2 = 11,
}
impl FUnaryOp {
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Frsqrt),
            1 => Some(Self::Frcp),
            2 => Some(Self::Ffloor),
            3 => Some(Self::Fceil),
            4 => Some(Self::Fround),
            5 => Some(Self::Ftrunc),
            6 => Some(Self::Ffract),
            7 => Some(Self::Fsat),
            8 => Some(Self::Fsin),
            9 => Some(Self::Fcos),
            10 => Some(Self::Fexp2),
            11 => Some(Self::Flog2),
            _ => None,
        }
    }
    #[must_use]
    pub fn mnemonic(self) -> &'static str {
        match self {
            Self::Frsqrt => "frsqrt",
            Self::Frcp => "frcp",
            Self::Ffloor => "ffloor",
            Self::Fceil => "fceil",
            Self::Fround => "fround",
            Self::Ftrunc => "ftrunc",
            Self::Ffract => "ffract",
            Self::Fsat => "fsat",
            Self::Fsin => "fsin",
            Self::Fcos => "fcos",
            Self::Fexp2 => "fexp2",
            Self::Flog2 => "flog2",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum F16Op {
    Hadd = 0,
    Hsub = 1,
    Hmul = 2,
    Hma = 3,
}
impl F16Op {
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Hadd),
            1 => Some(Self::Hsub),
            2 => Some(Self::Hmul),
            3 => Some(Self::Hma),
            _ => None,
        }
    }
    #[must_use]
    pub fn mnemonic(self) -> &'static str {
        match self {
            Self::Hadd => "hadd",
            Self::Hsub => "hsub",
            Self::Hmul => "hmul",
            Self::Hma => "hma",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum F16PackedOp {
    Hadd2 = 0,
    Hmul2 = 1,
    Hma2 = 2,
}
impl F16PackedOp {
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Hadd2),
            1 => Some(Self::Hmul2),
            2 => Some(Self::Hma2),
            _ => None,
        }
    }
    #[must_use]
    pub fn mnemonic(self) -> &'static str {
        match self {
            Self::Hadd2 => "hadd2",
            Self::Hmul2 => "hmul2",
            Self::Hma2 => "hma2",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum F64Op {
    Dadd = 0,
    Dsub = 1,
    Dmul = 2,
    Dma = 3,
}
impl F64Op {
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Dadd),
            1 => Some(Self::Dsub),
            2 => Some(Self::Dmul),
            3 => Some(Self::Dma),
            _ => None,
        }
    }
    #[must_use]
    pub fn mnemonic(self) -> &'static str {
        match self {
            Self::Dadd => "dadd",
            Self::Dsub => "dsub",
            Self::Dmul => "dmul",
            Self::Dma => "dma",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum F64DivSqrtOp {
    Ddiv = 0,
    Dsqrt = 1,
}
impl F64DivSqrtOp {
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Ddiv),
            1 => Some(Self::Dsqrt),
            _ => None,
        }
    }
    #[must_use]
    pub fn mnemonic(self) -> &'static str {
        match self {
            Self::Ddiv => "ddiv",
            Self::Dsqrt => "dsqrt",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BitOpType {
    Bitcount = 0,
    Bitfind = 1,
    Bitrev = 2,
    Bfe = 3,
    Bfi = 4,
}
impl BitOpType {
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Bitcount),
            1 => Some(Self::Bitfind),
            2 => Some(Self::Bitrev),
            3 => Some(Self::Bfe),
            4 => Some(Self::Bfi),
            _ => None,
        }
    }
    #[must_use]
    pub fn mnemonic(self) -> &'static str {
        match self {
            Self::Bitcount => "bitcount",
            Self::Bitfind => "bitfind",
            Self::Bitrev => "bitrev",
            Self::Bfe => "bfe",
            Self::Bfi => "bfi",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
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
impl CmpOp {
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Eq),
            1 => Some(Self::Ne),
            2 => Some(Self::Lt),
            3 => Some(Self::Le),
            4 => Some(Self::Gt),
            5 => Some(Self::Ge),
            6 => Some(Self::Ord),
            7 => Some(Self::Unord),
            _ => None,
        }
    }
    #[must_use]
    pub fn suffix(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
            Self::Ord => "ord",
            Self::Unord => "unord",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CvtType {
    F32I32 = 0,
    F32U32 = 1,
    I32F32 = 2,
    U32F32 = 3,
    F32F16 = 4,
    F16F32 = 5,
    F32F64 = 6,
    F64F32 = 7,
}
impl CvtType {
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::F32I32),
            1 => Some(Self::F32U32),
            2 => Some(Self::I32F32),
            3 => Some(Self::U32F32),
            4 => Some(Self::F32F16),
            5 => Some(Self::F16F32),
            6 => Some(Self::F32F64),
            7 => Some(Self::F64F32),
            _ => None,
        }
    }
    #[must_use]
    pub fn mnemonic(self) -> &'static str {
        match self {
            Self::F32I32 => "cvt_f32_i32",
            Self::F32U32 => "cvt_f32_u32",
            Self::I32F32 => "cvt_i32_f32",
            Self::U32F32 => "cvt_u32_f32",
            Self::F32F16 => "cvt_f32_f16",
            Self::F16F32 => "cvt_f16_f32",
            Self::F32F64 => "cvt_f32_f64",
            Self::F64F32 => "cvt_f64_f32",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MemWidth {
    U8 = 0,
    U16 = 1,
    U32 = 2,
    U64 = 3,
    U128 = 4,
}
impl MemWidth {
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::U8),
            1 => Some(Self::U16),
            2 => Some(Self::U32),
            3 => Some(Self::U64),
            4 => Some(Self::U128),
            _ => None,
        }
    }
    #[must_use]
    pub fn suffix(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::U128 => "u128",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
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
impl AtomicOp {
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Add),
            1 => Some(Self::Sub),
            2 => Some(Self::Min),
            3 => Some(Self::Max),
            4 => Some(Self::And),
            5 => Some(Self::Or),
            6 => Some(Self::Xor),
            7 => Some(Self::Exchange),
            _ => None,
        }
    }
    #[must_use]
    pub fn suffix(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Min => "min",
            Self::Max => "max",
            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
            Self::Exchange => "exchange",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Scope {
    Wave = 0,
    Workgroup = 1,
    Device = 2,
    System = 3,
}
impl Scope {
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Wave),
            1 => Some(Self::Workgroup),
            2 => Some(Self::Device),
            3 => Some(Self::System),
            _ => None,
        }
    }
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Wave => "wave",
            Self::Workgroup => "workgroup",
            Self::Device => "device",
            Self::System => "system",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
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
impl WaveOpType {
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Shuffle),
            1 => Some(Self::ShuffleUp),
            2 => Some(Self::ShuffleDown),
            3 => Some(Self::ShuffleXor),
            4 => Some(Self::Broadcast),
            5 => Some(Self::Ballot),
            6 => Some(Self::Any),
            7 => Some(Self::All),
            _ => None,
        }
    }
    #[must_use]
    pub fn mnemonic(self) -> &'static str {
        match self {
            Self::Shuffle => "wave_shuffle",
            Self::ShuffleUp => "wave_shuffle_up",
            Self::ShuffleDown => "wave_shuffle_down",
            Self::ShuffleXor => "wave_shuffle_xor",
            Self::Broadcast => "wave_broadcast",
            Self::Ballot => "wave_ballot",
            Self::Any => "wave_any",
            Self::All => "wave_all",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WaveReduceType {
    PrefixSum = 0,
    ReduceAdd = 1,
    ReduceMin = 2,
    ReduceMax = 3,
}
impl WaveReduceType {
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::PrefixSum),
            1 => Some(Self::ReduceAdd),
            2 => Some(Self::ReduceMin),
            3 => Some(Self::ReduceMax),
            _ => None,
        }
    }
    #[must_use]
    pub fn mnemonic(self) -> &'static str {
        match self {
            Self::PrefixSum => "wave_prefix_sum",
            Self::ReduceAdd => "wave_reduce_add",
            Self::ReduceMin => "wave_reduce_min",
            Self::ReduceMax => "wave_reduce_max",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
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
impl ControlOp {
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::If),
            1 => Some(Self::Else),
            2 => Some(Self::Endif),
            3 => Some(Self::Loop),
            4 => Some(Self::Break),
            5 => Some(Self::Continue),
            6 => Some(Self::Endloop),
            7 => Some(Self::Call),
            _ => None,
        }
    }
    #[must_use]
    pub fn mnemonic(self) -> &'static str {
        match self {
            Self::If => "if",
            Self::Else => "else",
            Self::Endif => "endif",
            Self::Loop => "loop",
            Self::Break => "break",
            Self::Continue => "continue",
            Self::Endloop => "endloop",
            Self::Call => "call",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
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
impl SyncOp {
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Return),
            1 => Some(Self::Halt),
            2 => Some(Self::Barrier),
            3 => Some(Self::FenceAcquire),
            4 => Some(Self::FenceRelease),
            5 => Some(Self::FenceAcqRel),
            6 => Some(Self::Wait),
            7 => Some(Self::Nop),
            _ => None,
        }
    }
    #[must_use]
    pub fn mnemonic(self) -> &'static str {
        match self {
            Self::Return => "return",
            Self::Halt => "halt",
            Self::Barrier => "barrier",
            Self::FenceAcquire => "fence_acquire",
            Self::FenceRelease => "fence_release",
            Self::FenceAcqRel => "fence_acq_rel",
            Self::Wait => "wait",
            Self::Nop => "nop",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MiscOp {
    Mov = 0,
    MovImm = 1,
    MovSr = 2,
}
impl MiscOp {
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Mov),
            1 => Some(Self::MovImm),
            2 => Some(Self::MovSr),
            _ => None,
        }
    }
    #[must_use]
    pub fn mnemonic(self) -> &'static str {
        match self {
            Self::Mov | Self::MovSr => "mov",
            Self::MovImm => "mov_imm",
        }
    }
}
pub const SPECIAL_REGISTER_NAMES: [&str; 16] = [
    "sr_thread_id_x",
    "sr_thread_id_y",
    "sr_thread_id_z",
    "sr_wave_id",
    "sr_lane_id",
    "sr_workgroup_id_x",
    "sr_workgroup_id_y",
    "sr_workgroup_id_z",
    "sr_workgroup_size_x",
    "sr_workgroup_size_y",
    "sr_workgroup_size_z",
    "sr_grid_size_x",
    "sr_grid_size_y",
    "sr_grid_size_z",
    "sr_wave_width",
    "sr_num_waves",
];
#[must_use]
pub fn special_register_name(index: u8) -> Option<&'static str> {
    SPECIAL_REGISTER_NAMES.get(index as usize).copied()
}
