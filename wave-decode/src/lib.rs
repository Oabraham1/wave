// Copyright (c) 2026 Ojima Abraham. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

// Shared instruction decoder for the WAVE ISA. Provides opcode definitions,
// instruction decoding from binary to structured types, and WBIN container
// format parsing. Used by wave-dis and wave-emu.

pub mod decoder;
pub mod instruction;
pub mod opcodes;
pub mod wbin;

pub use decoder::{decode_all, decode_at, DecodeError, Decoder};
pub use instruction::{DecodedInstruction, Operation};
pub use opcodes::{
    special_register_name, AtomicOp, BitOpType, CmpOp, ControlOp, CvtType, F16Op, F16PackedOp,
    F64DivSqrtOp, F64Op, FUnaryOp, MemWidth, MiscOp, Opcode, Scope, SyncOp, WaveOpType,
    WaveReduceType, SPECIAL_REGISTER_NAMES,
};
pub use wbin::{KernelInfo, WbinError, WbinFile, WbinHeader, WBIN_MAGIC, WBIN_VERSION};
