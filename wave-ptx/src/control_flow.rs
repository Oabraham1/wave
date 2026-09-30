// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Translate structured WAVE branches and loops into PTX control flow.

use crate::registers::pred;
use crate::CompileError;

#[derive(Debug)]
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

#[derive(Debug, Default)]
pub struct ControlFlowState {
    label_counter: u32,
    stack: Vec<Frame>,
}

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

    /// Emit PTX for else.
    ///
    /// # Errors
    ///
    /// Returns an error if the instruction has no matching block or repeats an else branch.
    pub fn emit_else(&mut self) -> Result<Vec<String>, CompileError> {
        let frame = self.stack.last_mut();
        let Some(Frame::If {
            else_label,
            endif_label,
            has_else,
        }) = frame
        else {
            return Err(CompileError::InvalidControlFlow(
                "else without matching block",
            ));
        };

        if *has_else {
            return Err(CompileError::InvalidControlFlow("duplicate else"));
        }
        *has_else = true;
        let jump = format!("bra.uni {endif_label};");
        let label = format!("{else_label}:");

        Ok(vec![jump, label])
    }

    /// Emit PTX for endif.
    ///
    /// # Errors
    ///
    /// Returns an error if the instruction has no matching if block.
    pub fn emit_endif(&mut self) -> Result<Vec<String>, CompileError> {
        let frame = self.stack.last();
        let Some(Frame::If {
            else_label,
            endif_label,
            has_else,
        }) = frame
        else {
            return Err(CompileError::InvalidControlFlow(
                "endif without matching block",
            ));
        };

        let lines = if *has_else {
            vec![format!("{endif_label}:")]
        } else {
            vec![format!("{else_label}:")]
        };
        self.stack.pop();
        Ok(lines)
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

    /// Emit PTX for break.
    ///
    /// # Errors
    ///
    /// Returns an error if the instruction is outside a loop.
    pub fn emit_break(&self, ps: u8) -> Result<Vec<String>, CompileError> {
        let frame = self.find_loop();
        let Some(Frame::Loop { endloop_label, .. }) = frame else {
            return Err(CompileError::InvalidControlFlow(
                "break without matching block",
            ));
        };

        Ok(vec![format!("@{} bra {};", pred(ps), endloop_label)])
    }

    /// Emit PTX for continue.
    ///
    /// # Errors
    ///
    /// Returns an error if the instruction is outside a loop.
    pub fn emit_continue(&self, ps: u8) -> Result<Vec<String>, CompileError> {
        let frame = self.find_loop();
        let Some(Frame::Loop { loop_label, .. }) = frame else {
            return Err(CompileError::InvalidControlFlow(
                "continue without matching block",
            ));
        };

        Ok(vec![format!("@{} bra {};", pred(ps), loop_label)])
    }

    /// Emit PTX for endloop.
    ///
    /// # Errors
    ///
    /// Returns an error if the instruction has no matching loop block.
    pub fn emit_endloop(&mut self) -> Result<Vec<String>, CompileError> {
        let frame = self.stack.last();
        let Some(Frame::Loop {
            loop_label,
            endloop_label,
        }) = frame
        else {
            return Err(CompileError::InvalidControlFlow(
                "endloop without matching block",
            ));
        };

        let lines = vec![
            format!("bra.uni {};", loop_label),
            format!("{endloop_label}:"),
        ];
        self.stack.pop();
        Ok(lines)
    }

    /// Check that all structured blocks have ended.
    ///
    /// # Errors
    ///
    /// Returns an error for an unterminated if or loop block.
    pub(crate) fn finish(&self) -> Result<(), CompileError> {
        match self.stack.last() {
            None => Ok(()),
            Some(Frame::If { .. }) => Err(CompileError::InvalidControlFlow("unterminated if")),
            Some(Frame::Loop { .. }) => Err(CompileError::InvalidControlFlow("unterminated loop")),
        }
    }

    fn find_loop(&self) -> Option<&Frame> {
        self.stack
            .iter()
            .rev()
            .find(|f| matches!(f, Frame::Loop { .. }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unfinished_blocks_return_errors() {
        let mut state = ControlFlowState::new();
        assert!(state.finish().is_ok());
        state.emit_loop();
        assert!(state.finish().is_err());
        state.emit_if(1);
        assert!(state.finish().is_err());
        state.emit_endif().unwrap();
        state.emit_endloop().unwrap();
        assert!(state.finish().is_ok());
    }

    #[test]
    fn unmatched_blocks_return_errors() {
        let mut state = ControlFlowState::new();
        assert!(state.emit_else().is_err());
        assert!(state.emit_endif().is_err());
        assert!(state.emit_break(0).is_err());
        assert!(state.emit_continue(0).is_err());
        assert!(state.emit_endloop().is_err());
        state.emit_if(0);
        assert!(state.emit_endloop().is_err());
        assert!(state.emit_endif().is_ok());
    }

    #[test]
    fn rejects_duplicate_else() {
        let mut state = ControlFlowState::new();
        state.emit_if(0);
        state.emit_else().unwrap();
        assert!(state.emit_else().is_err());
    }
}
