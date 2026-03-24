// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Control flow SYCL emission. Translates WAVE structured control flow instructions
//!
//! into equivalent C++ control flow statements. SYCL supports structured control
//! flow natively inside kernel lambdas so the mapping is direct, identical to the
//! Metal and HIP backends.

use crate::registers::pred;

#[must_use]
pub fn emit_if(ps: u8) -> String {
    format!("if ({}) {{", pred(ps))
}

#[must_use]
pub fn emit_else() -> &'static str {
    "} else {"
}

#[must_use]
pub fn emit_endif() -> &'static str {
    "}"
}

#[must_use]
pub fn emit_loop() -> &'static str {
    "while (true) {"
}

#[must_use]
pub fn emit_break(ps: u8) -> String {
    format!("if ({}) break;", pred(ps))
}

#[must_use]
pub fn emit_continue(ps: u8) -> String {
    format!("if ({}) continue;", pred(ps))
}

#[must_use]
pub fn emit_endloop() -> &'static str {
    "}"
}
