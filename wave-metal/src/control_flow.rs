// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Translate structured WAVE branches and loops into Metal control flow.

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
