// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Structured control flow with divergence support. Maintains a stack of active
//!
//! masks for nested if/else/endif and loop/break/endloop constructs. Divergence
//! occurs when threads disagree on predicates, executing both paths sequentially.

use crate::EmulatorError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlFrameKind {
    If,
    Loop,
}

#[derive(Debug, Clone)]
pub struct ControlFrame {
    pub kind: ControlFrameKind,
    pub entry_mask: u64,
    pub then_mask: u64,
    pub else_mask: u64,
    pub break_mask: u64,
    pub loop_start_pc: u32,
    pub in_else: bool,
}

impl ControlFrame {
    pub fn new_if(entry_mask: u64, then_mask: u64, else_mask: u64) -> Self {
        Self {
            kind: ControlFrameKind::If,
            entry_mask,
            then_mask,
            else_mask,
            break_mask: 0,
            loop_start_pc: 0,
            in_else: false,
        }
    }

    pub fn new_loop(entry_mask: u64, loop_start_pc: u32) -> Self {
        Self {
            kind: ControlFrameKind::Loop,
            entry_mask,
            then_mask: entry_mask,
            else_mask: 0,
            break_mask: 0,
            loop_start_pc,
            in_else: false,
        }
    }
}

#[derive(Debug)]
pub struct ControlFlowStack {
    frames: Vec<ControlFrame>,
    max_depth: usize,
}

impl ControlFlowStack {
    pub fn new(max_depth: usize) -> Self {
        Self {
            frames: Vec::with_capacity(max_depth),
            max_depth,
        }
    }

    pub fn push(&mut self, frame: ControlFrame) -> Result<(), EmulatorError> {
        if self.frames.len() >= self.max_depth {
            return Err(EmulatorError::StackOverflow {
                kind: "control flow".into(),
            });
        }
        self.frames.push(frame);
        Ok(())
    }

    pub fn pop(&mut self) -> Option<ControlFrame> {
        self.frames.pop()
    }

    pub fn top(&self) -> Option<&ControlFrame> {
        self.frames.last()
    }

    pub fn top_mut(&mut self) -> Option<&mut ControlFrame> {
        self.frames.last_mut()
    }

    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn find_loop(&self) -> Option<&ControlFrame> {
        self.frames
            .iter()
            .rev()
            .find(|f| f.kind == ControlFrameKind::Loop)
    }

    pub fn find_loop_mut(&mut self) -> Option<&mut ControlFrame> {
        self.frames
            .iter_mut()
            .rev()
            .find(|f| f.kind == ControlFrameKind::Loop)
    }
}

#[derive(Debug)]
pub struct ControlFlowManager {
    stack: ControlFlowStack,
}

impl ControlFlowManager {
    pub fn new() -> Self {
        Self {
            stack: ControlFlowStack::new(32),
        }
    }

    pub fn handle_if(
        &mut self,
        active_mask: u64,
        predicate_mask: u64,
    ) -> Result<(u64, Option<u32>), EmulatorError> {
        let then_mask = active_mask & predicate_mask;
        let else_mask = active_mask & !predicate_mask;

        let frame = ControlFrame::new_if(active_mask, then_mask, else_mask);
        self.stack.push(frame)?;

        Ok((then_mask, None))
    }

    pub fn handle_else(&mut self, _active_mask: u64) -> Result<(u64, Option<u32>), EmulatorError> {
        let frame = self
            .stack
            .top_mut()
            .ok_or_else(|| EmulatorError::ControlFlowError {
                message: "else without matching if".into(),
            })?;

        if frame.kind != ControlFrameKind::If {
            return Err(EmulatorError::ControlFlowError {
                message: "else without matching if".into(),
            });
        }

        frame.in_else = true;
        Ok((frame.else_mask, None))
    }

    pub fn handle_endif(&mut self) -> Result<u64, EmulatorError> {
        let frame = self
            .stack
            .pop()
            .ok_or_else(|| EmulatorError::ControlFlowError {
                message: "endif without matching if".into(),
            })?;

        if frame.kind != ControlFrameKind::If {
            self.stack.push(frame)?;
            return Err(EmulatorError::ControlFlowError {
                message: "endif without matching if".into(),
            });
        }

        Ok(frame.entry_mask)
    }

    pub fn handle_loop(&mut self, active_mask: u64, pc: u32) -> Result<u64, EmulatorError> {
        let frame = ControlFrame::new_loop(active_mask, pc);
        self.stack.push(frame)?;
        Ok(active_mask)
    }

    pub fn handle_break(
        &mut self,
        active_mask: u64,
        predicate_mask: u64,
    ) -> Result<(u64, Option<u32>), EmulatorError> {
        let frame = self
            .stack
            .find_loop_mut()
            .ok_or_else(|| EmulatorError::ControlFlowError {
                message: "break outside of loop".into(),
            })?;

        let breaking_threads = active_mask & predicate_mask;
        frame.break_mask |= breaking_threads;

        let new_mask = active_mask & !breaking_threads;

        if new_mask == 0 {
            Ok((new_mask, Some(0)))
        } else {
            Ok((new_mask, None))
        }
    }

    pub fn handle_continue(
        &mut self,
        active_mask: u64,
        predicate_mask: u64,
    ) -> Result<(u64, Option<u32>), EmulatorError> {
        let frame = self
            .stack
            .find_loop()
            .ok_or_else(|| EmulatorError::ControlFlowError {
                message: "continue outside of loop".into(),
            })?;

        let continuing_threads = active_mask & predicate_mask;
        let new_mask = active_mask & !continuing_threads;

        if new_mask == 0 || continuing_threads != 0 {
            Ok((new_mask, Some(frame.loop_start_pc)))
        } else {
            Ok((new_mask, None))
        }
    }

    pub fn handle_endloop(
        &mut self,
        _active_mask: u64,
    ) -> Result<(u64, Option<u32>), EmulatorError> {

        let frame = self
            .stack
            .top_mut()
            .ok_or_else(|| EmulatorError::ControlFlowError {
                message: "endloop without matching loop".into(),
            })?;

        if frame.kind != ControlFrameKind::Loop {
            return Err(EmulatorError::ControlFlowError {
                message: "endloop without matching loop".into(),
            });
        }

        let remaining_mask = frame.entry_mask & !frame.break_mask;

        if remaining_mask != 0 {
            let loop_start = frame.loop_start_pc;
            Ok((remaining_mask, Some(loop_start)))
        } else {
            let frame = self
                .stack
                .pop()
                .ok_or_else(|| EmulatorError::ControlFlowError {
                    message: "endloop without matching loop".into(),
                })?;
            Ok((frame.entry_mask, None))
        }
    }

    pub fn pop_loop(&mut self) -> Result<u64, EmulatorError> {
        let frame = self
            .stack
            .pop()
            .ok_or_else(|| EmulatorError::ControlFlowError {
                message: "endloop without matching loop".into(),
            })?;

        if frame.kind != ControlFrameKind::Loop {
            self.stack.push(frame)?;
            return Err(EmulatorError::ControlFlowError {
                message: "endloop without matching loop".into(),
            });
        }

        Ok(frame.entry_mask & !frame.break_mask)
    }

    pub fn depth(&self) -> usize {
        self.stack.depth()
    }

    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    pub fn current_loop_start(&self) -> Option<u32> {
        self.stack.find_loop().map(|f| f.loop_start_pc)
    }
}

impl Default for ControlFlowManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cf_simple_if() {
        let mut cf = ControlFlowManager::new();

        let (mask, _) = cf.handle_if(0xFFFF_FFFF, 0x0000_FFFF).unwrap();
        assert_eq!(mask, 0x0000_FFFF);

        let (mask, _) = cf.handle_else(mask).unwrap();
        assert_eq!(mask, 0xFFFF_0000);

        let mask = cf.handle_endif().unwrap();
        assert_eq!(mask, 0xFFFF_FFFF);
    }

    #[test]
    fn test_cf_if_no_else_needed() {
        let mut cf = ControlFlowManager::new();

        let (mask, _) = cf.handle_if(0xFFFF_FFFF, 0xFFFF_FFFF).unwrap();
        assert_eq!(mask, 0xFFFF_FFFF);

        let mask = cf.handle_endif().unwrap();
        assert_eq!(mask, 0xFFFF_FFFF);
    }

    #[test]
    fn test_cf_if_all_take_else() {
        let mut cf = ControlFlowManager::new();

        let (then_mask, _) = cf.handle_if(0xFFFF_FFFF, 0).unwrap();
        assert_eq!(then_mask, 0);

        let (else_mask, _) = cf.handle_else(then_mask).unwrap();
        assert_eq!(else_mask, 0xFFFF_FFFF);

        let mask = cf.handle_endif().unwrap();
        assert_eq!(mask, 0xFFFF_FFFF);
    }

    #[test]
    fn test_cf_simple_loop() {
        let mut cf = ControlFlowManager::new();

        let mask = cf.handle_loop(0xFFFF_FFFF, 0x100).unwrap();
        assert_eq!(mask, 0xFFFF_FFFF);
        assert_eq!(cf.current_loop_start(), Some(0x100));

        let (mask, jump) = cf.handle_break(mask, 0x0000_FFFF).unwrap();
        assert_eq!(mask, 0xFFFF_0000);
        assert!(jump.is_none());
    }

    #[test]
    fn test_cf_loop_all_break() {
        let mut cf = ControlFlowManager::new();

        cf.handle_loop(0xFFFF_FFFF, 0x100).unwrap();
        let (mask, jump) = cf.handle_break(0xFFFF_FFFF, 0xFFFF_FFFF).unwrap();
        assert_eq!(mask, 0);
        assert!(jump.is_some());
    }

    #[test]
    fn test_cf_nested_if() {
        let mut cf = ControlFlowManager::new();

        let (mask1, _) = cf.handle_if(0xFFFF_FFFF, 0x00FF_00FF).unwrap();
        assert_eq!(mask1, 0x00FF_00FF);

        let (mask2, _) = cf.handle_if(mask1, 0x0000_00FF).unwrap();
        assert_eq!(mask2, 0x0000_00FF);

        let mask2 = cf.handle_endif().unwrap();
        assert_eq!(mask2, 0x00FF_00FF);

        let mask1 = cf.handle_endif().unwrap();
        assert_eq!(mask1, 0xFFFF_FFFF);
    }

    #[test]
    fn test_cf_stack_overflow() {
        let mut cf = ControlFlowManager::new();

        for i in 0..32 {
            cf.handle_if(0xFFFF_FFFF, 0x0000_FFFF).unwrap();
            assert_eq!(cf.depth(), i + 1);
        }

        assert!(cf.handle_if(0xFFFF_FFFF, 0x0000_FFFF).is_err());
    }

    #[test]
    fn test_cf_mismatched_endif() {
        let mut cf = ControlFlowManager::new();
        assert!(cf.handle_endif().is_err());
    }

    #[test]
    fn test_cf_break_outside_loop() {
        let mut cf = ControlFlowManager::new();
        assert!(cf.handle_break(0xFFFF_FFFF, 0xFFFF_FFFF).is_err());
    }
}
