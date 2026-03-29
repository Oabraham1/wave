// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Opcode definitions and instruction signature tables. Maps mnemonic strings to
//!
//! instruction encoding metadata (opcode, operand types, modifiers). Uses `LazyLock`
//! for static initialization of the lookup tables.

use std::collections::HashMap;
use std::sync::LazyLock;

pub const OPCODE_SHIFT: u32 = 26;
pub const OPCODE_MASK: u32 = 0x3F;
pub const RD_SHIFT: u32 = 21;
pub const RD_MASK: u32 = 0x1F;
pub const RS1_SHIFT: u32 = 16;
pub const RS1_MASK: u32 = 0x1F;
pub const RS2_SHIFT: u32 = 11;
pub const RS2_MASK: u32 = 0x1F;
pub const MODIFIER_SHIFT: u32 = 7;
pub const MODIFIER_MASK: u32 = 0x0F;
pub const SCOPE_SHIFT: u32 = 5;
pub const SCOPE_MASK: u32 = 0x03;
pub const PRED_SHIFT: u32 = 3;
pub const PRED_MASK: u32 = 0x03;
pub const PRED_NEG_SHIFT: u32 = 2;
pub const PRED_NEG_MASK: u32 = 0x01;
pub const FLAGS_SHIFT: u32 = 0;
pub const FLAGS_MASK: u32 = 0x03;

pub const EXTENDED_RS3_SHIFT: u32 = 27;
pub const EXTENDED_RS3_MASK: u32 = 0x1F;

pub const MAX_REGISTER_INDEX: u8 = 31;

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
    pub is_sync: bool,
    pub is_misc: bool,
    pub wave_reduce: bool,
}

pub static MNEMONIC_MAP: LazyLock<HashMap<&'static str, InstructionSignature>> =
    LazyLock::new(build_mnemonic_map);

fn build_mnemonic_map() -> HashMap<&'static str, InstructionSignature> {
    use OperandKind::{Imm32, Label, OptionalRd, Pd, PdSrc, Rd, Rs1, Rs2, Rs3, Rs4, SpecialReg};

    let mut m = HashMap::new();

    let rrr: &[OperandKind] = &[Rd, Rs1, Rs2];
    let rr: &[OperandKind] = &[Rd, Rs1];
    let rrrr: &[OperandKind] = &[Rd, Rs1, Rs2, Rs3];

    macro_rules! base {
        ($name:expr, $op:expr, $operands:expr) => {
            m.insert(
                $name,
                InstructionSignature {
                    opcode: $op,
                    operands: $operands,
                    modifier: None,
                    extended: false,
                    is_sync: false,
                    is_misc: false,
                    wave_reduce: false,
                },
            );
        };
        ($name:expr, $op:expr, $operands:expr, $mod:expr) => {
            m.insert(
                $name,
                InstructionSignature {
                    opcode: $op,
                    operands: $operands,
                    modifier: Some($mod),
                    extended: false,
                    is_sync: false,
                    is_misc: false,
                    wave_reduce: false,
                },
            );
        };
    }

    macro_rules! extended {
        ($name:expr, $op:expr, $operands:expr) => {
            m.insert(
                $name,
                InstructionSignature {
                    opcode: $op,
                    operands: $operands,
                    modifier: None,
                    extended: true,
                    is_sync: false,
                    is_misc: false,
                    wave_reduce: false,
                },
            );
        };
        ($name:expr, $op:expr, $operands:expr, $mod:expr) => {
            m.insert(
                $name,
                InstructionSignature {
                    opcode: $op,
                    operands: $operands,
                    modifier: Some($mod),
                    extended: true,
                    is_sync: false,
                    is_misc: false,
                    wave_reduce: false,
                },
            );
        };
    }

    base!("iadd", Opcode::Iadd, rrr);
    base!("isub", Opcode::Isub, rrr);
    base!("imul", Opcode::Imul, rrr);
    base!("imul_hi", Opcode::ImulHi, rrr);
    extended!("imad", Opcode::Imad, rrrr);
    base!("idiv", Opcode::Idiv, rrr);
    base!("imod", Opcode::Imod, rrr);
    base!("ineg", Opcode::Ineg, rr);
    base!("iabs", Opcode::Iabs, rr);
    base!("imin", Opcode::Imin, rrr);
    base!("imax", Opcode::Imax, rrr);
    extended!("iclamp", Opcode::Iclamp, rrrr);

    base!("fadd", Opcode::Fadd, rrr);
    base!("fsub", Opcode::Fsub, rrr);
    base!("fmul", Opcode::Fmul, rrr);
    extended!("fma", Opcode::Fma, rrrr);
    base!("fdiv", Opcode::Fdiv, rrr);
    base!("fneg", Opcode::Fneg, rr);
    base!("fabs", Opcode::Fabs, rr);
    base!("fmin", Opcode::Fmin, rrr);
    base!("fmax", Opcode::Fmax, rrr);
    extended!("fclamp", Opcode::Fclamp, rrrr);
    base!("fsqrt", Opcode::Fsqrt, rr);

    base!("frsqrt", Opcode::FUnaryOps, rr, FUnaryOp::Frsqrt as u8);
    base!("frcp", Opcode::FUnaryOps, rr, FUnaryOp::Frcp as u8);
    base!("ffloor", Opcode::FUnaryOps, rr, FUnaryOp::Ffloor as u8);
    base!("fceil", Opcode::FUnaryOps, rr, FUnaryOp::Fceil as u8);
    base!("fround", Opcode::FUnaryOps, rr, FUnaryOp::Fround as u8);
    base!("ftrunc", Opcode::FUnaryOps, rr, FUnaryOp::Ftrunc as u8);
    base!("ffract", Opcode::FUnaryOps, rr, FUnaryOp::Ffract as u8);
    base!("fsat", Opcode::FUnaryOps, rr, FUnaryOp::Fsat as u8);

    m.insert(
        "fsin",
        InstructionSignature {
            opcode: Opcode::FUnaryOps,
            operands: rr,
            modifier: Some(8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "fcos",
        InstructionSignature {
            opcode: Opcode::FUnaryOps,
            operands: rr,
            modifier: Some(9),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "fexp2",
        InstructionSignature {
            opcode: Opcode::FUnaryOps,
            operands: rr,
            modifier: Some(10),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "flog2",
        InstructionSignature {
            opcode: Opcode::FUnaryOps,
            operands: rr,
            modifier: Some(11),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );

    base!("hadd", Opcode::F16Ops, rrr, F16Op::Hadd as u8);
    base!("hsub", Opcode::F16Ops, rrr, F16Op::Hsub as u8);
    base!("hmul", Opcode::F16Ops, rrr, F16Op::Hmul as u8);
    extended!("hma", Opcode::F16Ops, rrrr, F16Op::Hma as u8);

    base!("hadd2", Opcode::F16PackedOps, rrr, F16PackedOp::Hadd2 as u8);
    base!("hmul2", Opcode::F16PackedOps, rrr, F16PackedOp::Hmul2 as u8);
    extended!("hma2", Opcode::F16PackedOps, rrrr, F16PackedOp::Hma2 as u8);

    base!("dadd", Opcode::F64Ops, rrr, F64Op::Dadd as u8);
    base!("dsub", Opcode::F64Ops, rrr, F64Op::Dsub as u8);
    base!("dmul", Opcode::F64Ops, rrr, F64Op::Dmul as u8);
    extended!("dma", Opcode::F64Ops, rrrr, F64Op::Dma as u8);
    base!("ddiv", Opcode::F64DivSqrt, rrr, F64DivSqrtOp::Ddiv as u8);
    base!("dsqrt", Opcode::F64DivSqrt, rr, F64DivSqrtOp::Dsqrt as u8);

    base!("and", Opcode::And, rrr);
    base!("or", Opcode::Or, rrr);
    base!("xor", Opcode::Xor, rrr);
    base!("not", Opcode::Not, rr);
    base!("shl", Opcode::Shl, rrr);
    base!("shr", Opcode::Shr, rrr);
    base!("sar", Opcode::Sar, rrr);

    base!("bitcount", Opcode::BitOps, rr, BitOpType::Bitcount as u8);
    base!("bitfind", Opcode::BitOps, rr, BitOpType::Bitfind as u8);
    base!("bitrev", Opcode::BitOps, rr, BitOpType::Bitrev as u8);
    extended!("bfe", Opcode::BitOps, rrrr, BitOpType::Bfe as u8);
    m.insert(
        "bfi",
        InstructionSignature {
            opcode: Opcode::BitOps,
            operands: &[Rd, Rs1, Rs2, Rs3, Rs4],
            modifier: Some(BitOpType::Bfi as u8),
            extended: true,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );

    base!("icmp_eq", Opcode::Icmp, &[Pd, Rs1, Rs2], CmpOp::Eq as u8);
    base!("icmp_ne", Opcode::Icmp, &[Pd, Rs1, Rs2], CmpOp::Ne as u8);
    base!("icmp_lt", Opcode::Icmp, &[Pd, Rs1, Rs2], CmpOp::Lt as u8);
    base!("icmp_le", Opcode::Icmp, &[Pd, Rs1, Rs2], CmpOp::Le as u8);
    base!("icmp_gt", Opcode::Icmp, &[Pd, Rs1, Rs2], CmpOp::Gt as u8);
    base!("icmp_ge", Opcode::Icmp, &[Pd, Rs1, Rs2], CmpOp::Ge as u8);

    base!("ucmp_lt", Opcode::Ucmp, &[Pd, Rs1, Rs2], CmpOp::Lt as u8);
    base!("ucmp_le", Opcode::Ucmp, &[Pd, Rs1, Rs2], CmpOp::Le as u8);

    base!("fcmp_eq", Opcode::Fcmp, &[Pd, Rs1, Rs2], CmpOp::Eq as u8);
    base!("fcmp_ne", Opcode::Fcmp, &[Pd, Rs1, Rs2], CmpOp::Ne as u8);
    base!("fcmp_lt", Opcode::Fcmp, &[Pd, Rs1, Rs2], CmpOp::Lt as u8);
    base!("fcmp_le", Opcode::Fcmp, &[Pd, Rs1, Rs2], CmpOp::Le as u8);
    base!("fcmp_gt", Opcode::Fcmp, &[Pd, Rs1, Rs2], CmpOp::Gt as u8);
    base!("fcmp_ord", Opcode::Fcmp, &[Pd, Rs1, Rs2], CmpOp::Ord as u8);
    base!(
        "fcmp_unord",
        Opcode::Fcmp,
        &[Pd, Rs1, Rs2],
        CmpOp::Unord as u8
    );

    base!("select", Opcode::Select, &[Rd, PdSrc, Rs1, Rs2]);

    base!("cvt_f32_i32", Opcode::Cvt, rr, CvtType::F32I32 as u8);
    base!("cvt_f32_u32", Opcode::Cvt, rr, CvtType::F32U32 as u8);
    base!("cvt_i32_f32", Opcode::Cvt, rr, CvtType::I32F32 as u8);
    base!("cvt_u32_f32", Opcode::Cvt, rr, CvtType::U32F32 as u8);
    base!("cvt_f32_f16", Opcode::Cvt, rr, CvtType::F32F16 as u8);
    base!("cvt_f16_f32", Opcode::Cvt, rr, CvtType::F16F32 as u8);
    base!("cvt_f32_f64", Opcode::Cvt, rr, CvtType::F32F64 as u8);
    base!("cvt_f64_f32", Opcode::Cvt, rr, CvtType::F64F32 as u8);

    base!("local_load_u8", Opcode::LocalLoad, rr, MemWidth::U8 as u8);
    base!("local_load_u16", Opcode::LocalLoad, rr, MemWidth::U16 as u8);
    base!("local_load_u32", Opcode::LocalLoad, rr, MemWidth::U32 as u8);
    base!("local_load_u64", Opcode::LocalLoad, rr, MemWidth::U64 as u8);

    base!(
        "local_store_u8",
        Opcode::LocalStore,
        &[Rs1, Rs2],
        MemWidth::U8 as u8
    );
    base!(
        "local_store_u16",
        Opcode::LocalStore,
        &[Rs1, Rs2],
        MemWidth::U16 as u8
    );
    base!(
        "local_store_u32",
        Opcode::LocalStore,
        &[Rs1, Rs2],
        MemWidth::U32 as u8
    );
    base!(
        "local_store_u64",
        Opcode::LocalStore,
        &[Rs1, Rs2],
        MemWidth::U64 as u8
    );

    base!("device_load_u8", Opcode::DeviceLoad, rr, MemWidth::U8 as u8);
    base!(
        "device_load_u16",
        Opcode::DeviceLoad,
        rr,
        MemWidth::U16 as u8
    );
    base!(
        "device_load_u32",
        Opcode::DeviceLoad,
        rr,
        MemWidth::U32 as u8
    );
    base!(
        "device_load_u64",
        Opcode::DeviceLoad,
        rr,
        MemWidth::U64 as u8
    );
    base!(
        "device_load_u128",
        Opcode::DeviceLoad,
        rr,
        MemWidth::U128 as u8
    );

    base!(
        "device_store_u8",
        Opcode::DeviceStore,
        &[Rs1, Rs2],
        MemWidth::U8 as u8
    );
    base!(
        "device_store_u16",
        Opcode::DeviceStore,
        &[Rs1, Rs2],
        MemWidth::U16 as u8
    );
    base!(
        "device_store_u32",
        Opcode::DeviceStore,
        &[Rs1, Rs2],
        MemWidth::U32 as u8
    );
    base!(
        "device_store_u64",
        Opcode::DeviceStore,
        &[Rs1, Rs2],
        MemWidth::U64 as u8
    );
    base!(
        "device_store_u128",
        Opcode::DeviceStore,
        &[Rs1, Rs2],
        MemWidth::U128 as u8
    );

    m.insert(
        "local_atomic_add",
        InstructionSignature {
            opcode: Opcode::LocalAtomic,
            operands: &[OptionalRd, Rs1, Rs2],
            modifier: Some(AtomicOp::Add as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "local_atomic_sub",
        InstructionSignature {
            opcode: Opcode::LocalAtomic,
            operands: &[OptionalRd, Rs1, Rs2],
            modifier: Some(AtomicOp::Sub as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "local_atomic_min",
        InstructionSignature {
            opcode: Opcode::LocalAtomic,
            operands: &[OptionalRd, Rs1, Rs2],
            modifier: Some(AtomicOp::Min as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "local_atomic_max",
        InstructionSignature {
            opcode: Opcode::LocalAtomic,
            operands: &[OptionalRd, Rs1, Rs2],
            modifier: Some(AtomicOp::Max as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "local_atomic_and",
        InstructionSignature {
            opcode: Opcode::LocalAtomic,
            operands: &[OptionalRd, Rs1, Rs2],
            modifier: Some(AtomicOp::And as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "local_atomic_or",
        InstructionSignature {
            opcode: Opcode::LocalAtomic,
            operands: &[OptionalRd, Rs1, Rs2],
            modifier: Some(AtomicOp::Or as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "local_atomic_xor",
        InstructionSignature {
            opcode: Opcode::LocalAtomic,
            operands: &[OptionalRd, Rs1, Rs2],
            modifier: Some(AtomicOp::Xor as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "local_atomic_exchange",
        InstructionSignature {
            opcode: Opcode::LocalAtomic,
            operands: &[OptionalRd, Rs1, Rs2],
            modifier: Some(AtomicOp::Exchange as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "local_atomic_cas",
        InstructionSignature {
            opcode: Opcode::LocalAtomic,
            operands: &[OptionalRd, Rs1, Rs2, Rs3],
            modifier: None,
            extended: true,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );

    m.insert(
        "atomic_add",
        InstructionSignature {
            opcode: Opcode::DeviceAtomic,
            operands: &[OptionalRd, Rs1, Rs2, OperandKind::Scope],
            modifier: Some(AtomicOp::Add as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "atomic_sub",
        InstructionSignature {
            opcode: Opcode::DeviceAtomic,
            operands: &[OptionalRd, Rs1, Rs2, OperandKind::Scope],
            modifier: Some(AtomicOp::Sub as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "atomic_min",
        InstructionSignature {
            opcode: Opcode::DeviceAtomic,
            operands: &[OptionalRd, Rs1, Rs2, OperandKind::Scope],
            modifier: Some(AtomicOp::Min as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "atomic_max",
        InstructionSignature {
            opcode: Opcode::DeviceAtomic,
            operands: &[OptionalRd, Rs1, Rs2, OperandKind::Scope],
            modifier: Some(AtomicOp::Max as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "atomic_and",
        InstructionSignature {
            opcode: Opcode::DeviceAtomic,
            operands: &[OptionalRd, Rs1, Rs2, OperandKind::Scope],
            modifier: Some(AtomicOp::And as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "atomic_or",
        InstructionSignature {
            opcode: Opcode::DeviceAtomic,
            operands: &[OptionalRd, Rs1, Rs2, OperandKind::Scope],
            modifier: Some(AtomicOp::Or as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "atomic_xor",
        InstructionSignature {
            opcode: Opcode::DeviceAtomic,
            operands: &[OptionalRd, Rs1, Rs2, OperandKind::Scope],
            modifier: Some(AtomicOp::Xor as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "atomic_exchange",
        InstructionSignature {
            opcode: Opcode::DeviceAtomic,
            operands: &[OptionalRd, Rs1, Rs2, OperandKind::Scope],
            modifier: Some(AtomicOp::Exchange as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "atomic_cas",
        InstructionSignature {
            opcode: Opcode::DeviceAtomic,
            operands: &[OptionalRd, Rs1, Rs2, Rs3, OperandKind::Scope],
            modifier: None,
            extended: true,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );

    base!(
        "wave_shuffle",
        Opcode::WaveOp,
        rrr,
        WaveOpType::Shuffle as u8
    );
    base!(
        "wave_shuffle_up",
        Opcode::WaveOp,
        rrr,
        WaveOpType::ShuffleUp as u8
    );
    base!(
        "wave_shuffle_down",
        Opcode::WaveOp,
        rrr,
        WaveOpType::ShuffleDown as u8
    );
    base!(
        "wave_shuffle_xor",
        Opcode::WaveOp,
        rrr,
        WaveOpType::ShuffleXor as u8
    );
    base!(
        "wave_broadcast",
        Opcode::WaveOp,
        rrr,
        WaveOpType::Broadcast as u8
    );
    base!(
        "wave_ballot",
        Opcode::WaveOp,
        &[Rd, PdSrc],
        WaveOpType::Ballot as u8
    );
    base!(
        "wave_any",
        Opcode::WaveOp,
        &[Pd, PdSrc],
        WaveOpType::Any as u8
    );
    base!(
        "wave_all",
        Opcode::WaveOp,
        &[Pd, PdSrc],
        WaveOpType::All as u8
    );

    m.insert(
        "wave_prefix_sum",
        InstructionSignature {
            opcode: Opcode::WaveOp,
            operands: rr,
            modifier: Some(WaveReduceType::PrefixSum as u8 + 8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "wave_reduce_add",
        InstructionSignature {
            opcode: Opcode::WaveOp,
            operands: rr,
            modifier: Some(WaveReduceType::ReduceAdd as u8 + 8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "wave_reduce_min",
        InstructionSignature {
            opcode: Opcode::WaveOp,
            operands: rr,
            modifier: Some(WaveReduceType::ReduceMin as u8 + 8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "wave_reduce_max",
        InstructionSignature {
            opcode: Opcode::WaveOp,
            operands: rr,
            modifier: Some(WaveReduceType::ReduceMax as u8 + 8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );

    m.insert(
        "if",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[PdSrc],
            modifier: Some(ControlOp::If as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "else",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[],
            modifier: Some(ControlOp::Else as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "endif",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[],
            modifier: Some(ControlOp::Endif as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "loop",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[],
            modifier: Some(ControlOp::Loop as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "break",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[PdSrc],
            modifier: Some(ControlOp::Break as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "continue",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[PdSrc],
            modifier: Some(ControlOp::Continue as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "endloop",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[],
            modifier: Some(ControlOp::Endloop as u8),
            extended: false,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "call",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[Label],
            modifier: Some(ControlOp::Call as u8),
            extended: true,
            is_sync: false,
            is_misc: false,
            wave_reduce: false,
        },
    );

    m.insert(
        "return",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[],
            modifier: Some(SyncOp::Return as u8),
            extended: false,
            is_sync: true,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "halt",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[],
            modifier: Some(SyncOp::Halt as u8),
            extended: false,
            is_sync: true,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "barrier",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[],
            modifier: Some(SyncOp::Barrier as u8),
            extended: false,
            is_sync: true,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "fence_acquire",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[OperandKind::Scope],
            modifier: Some(SyncOp::FenceAcquire as u8),
            extended: false,
            is_sync: true,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "fence_release",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[OperandKind::Scope],
            modifier: Some(SyncOp::FenceRelease as u8),
            extended: false,
            is_sync: true,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "fence_acq_rel",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[OperandKind::Scope],
            modifier: Some(SyncOp::FenceAcqRel as u8),
            extended: false,
            is_sync: true,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "wait",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[],
            modifier: Some(SyncOp::Wait as u8),
            extended: false,
            is_sync: true,
            is_misc: false,
            wave_reduce: false,
        },
    );
    m.insert(
        "nop",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[],
            modifier: Some(SyncOp::Nop as u8),
            extended: false,
            is_sync: true,
            is_misc: false,
            wave_reduce: false,
        },
    );

    m.insert(
        "mov",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[Rd, Rs1],
            modifier: Some(MiscOp::Mov as u8),
            extended: false,
            is_sync: false,
            is_misc: true,
            wave_reduce: false,
        },
    );
    m.insert(
        "mov_imm",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[Rd, Imm32],
            modifier: Some(MiscOp::MovImm as u8),
            extended: true,
            is_sync: false,
            is_misc: true,
            wave_reduce: false,
        },
    );

    m.insert(
        "mov_sr",
        InstructionSignature {
            opcode: Opcode::Control,
            operands: &[Rd, SpecialReg],
            modifier: Some(MiscOp::MovSr as u8),
            extended: false,
            is_sync: false,
            is_misc: true,
            wave_reduce: false,
        },
    );

    m
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

pub fn encode_base(
    opcode: u8,
    rd: u8,
    rs1: u8,
    rs2: u8,
    modifier: u8,
    scope: u8,
    pred: u8,
    pred_neg: bool,
    flags: u8,
) -> u32 {
    ((u32::from(opcode) & OPCODE_MASK) << OPCODE_SHIFT)
        | ((u32::from(rd) & RD_MASK) << RD_SHIFT)
        | ((u32::from(rs1) & RS1_MASK) << RS1_SHIFT)
        | ((u32::from(rs2) & RS2_MASK) << RS2_SHIFT)
        | ((u32::from(modifier) & MODIFIER_MASK) << MODIFIER_SHIFT)
        | ((u32::from(scope) & SCOPE_MASK) << SCOPE_SHIFT)
        | ((u32::from(pred) & PRED_MASK) << PRED_SHIFT)
        | (u32::from(pred_neg) << PRED_NEG_SHIFT)
        | (u32::from(flags) & FLAGS_MASK)
}
