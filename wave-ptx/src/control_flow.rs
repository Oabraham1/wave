// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Structured control flow lowering to PTX branches and labels. Converts WAVE's
//!
//! if/else/endif and loop/break/continue/endloop into predicated branches with
//! generated labels. Uses a stack of frames to track nesting and a monotonic
//! counter for unique label generation. Labels use the $L_ prefix convention.

use crate::registers::pred;

enum Frame {
    If {
        else_label: String,
        endif_label: String,
        has_else: bool,
    },
    Loop {
        loop_label: String,
        endloop_label: String,
    },
}

#[derive(Default)]
pub struct ControlFlowState {
    label_counter: u32,
    stack: Vec<Frame>,
}

#[allow(clippy::missing_panics_doc)]
impl ControlFlowState {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn emit_if(&mut self, ps: u8) -> Vec<String> {
        let n = self.label_counter;
        self.label_counter += 1;

        let else_label = format!("$L_else_{n}");
        let endif_label = format!("$L_endif_{n}");

        let branch = format!("@!{} bra {};", pred(ps), else_label);

        self.stack.push(Frame::If {
            else_label,
            endif_label,
            has_else: false,
        });

        vec![branch]
    }

    pub fn emit_else(&mut self) -> Vec<String> {
        let frame = self.stack.last_mut().expect("else without matching if");
        let Frame::If {
            else_label,
            endif_label,
            has_else,
        } = frame
        else {
            panic!("else without matching if");
        };

        *has_else = true;
        let jump = format!("bra.uni {endif_label};");
        let label = format!("{else_label}:");

        vec![jump, label]
    }

    pub fn emit_endif(&mut self) -> Vec<String> {
        let frame = self.stack.pop().expect("endif without matching if");
        let Frame::If {
            else_label,
            endif_label,
            has_else,
        } = frame
        else {
            panic!("endif without matching if");
        };

        if has_else {
            vec![format!("{endif_label}:")]
        } else {
            vec![format!("{else_label}:")]
        }
    }

    pub fn emit_loop(&mut self) -> Vec<String> {
        let n = self.label_counter;
        self.label_counter += 1;

        let loop_label = format!("$L_loop_{n}");
        let endloop_label = format!("$L_endloop_{n}");

        let label = format!("{loop_label}:");

        self.stack.push(Frame::Loop {
            loop_label,
            endloop_label,
        });

        vec![label]
    }

    #[must_use]
    pub fn emit_break(&self, ps: u8) -> Vec<String> {
        let frame = self.find_loop();
        let Frame::Loop { endloop_label, .. } = frame else {
            panic!("break without matching loop");
        };

        vec![format!("@{} bra {};", pred(ps), endloop_label)]
    }

    #[must_use]
    pub fn emit_continue(&self, ps: u8) -> Vec<String> {
        let frame = self.find_loop();
        let Frame::Loop { loop_label, .. } = frame else {
            panic!("continue without matching loop");
        };

        vec![format!("@{} bra {};", pred(ps), loop_label)]
    }

    pub fn emit_endloop(&mut self) -> Vec<String> {
        let frame = self.stack.pop().expect("endloop without matching loop");
        let Frame::Loop {
            loop_label,
            endloop_label,
        } = frame
        else {
            panic!("endloop without matching loop");
        };

        vec![
            format!("bra.uni {};", loop_label),
            format!("{endloop_label}:"),
        ]
    }

    fn find_loop(&self) -> &Frame {
        self.stack
            .iter()
            .rev()
            .find(|f| matches!(f, Frame::Loop { .. }))
            .expect("break/continue without enclosing loop")
    }
}
