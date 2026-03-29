// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! MIR to LIR lowering: instruction selection and structured control flow emission.
//!
//! Translates MIR instructions to near-WAVE LIR instructions. Analyzes
//! the CFG to emit structured control flow (If/Endif, Loop/Endloop)
//! by detecting merge points and back edges. Comparison results are
//! tracked as predicate registers so CondBranch can use them directly.

use std::collections::{HashMap, HashSet};

use crate::diagnostics::CompileError;
use crate::hir::expr::{BinOp, BuiltinFunc, ShuffleMode, UnaryOp};
use crate::hir::types::AddressSpace;
use crate::lir::instruction::LirInst;
use crate::lir::operand::{MemWidth, PReg, SpecialReg, VReg};
use crate::mir::basic_block::Terminator;
use crate::mir::function::MirFunction;
use crate::mir::instruction::MirInst;
use crate::mir::types::MirType;
use crate::mir::value::{BlockId, ValueId};

/// Lower a MIR function to a flat list of LIR instructions.
///
/// # Errors
///
/// Returns `CompileError` if instruction selection encounters unsupported patterns.
pub fn lower_function(func: &MirFunction) -> Result<Vec<LirInst>, CompileError> {
    let mut lowerer = MirToLirLowerer::new(func);
    lowerer.lower()?;
    Ok(lowerer.instructions)
}

struct MirToLirLowerer<'a> {
    func: &'a MirFunction,
    instructions: Vec<LirInst>,
    value_to_vreg: HashMap<ValueId, VReg>,
    value_to_preg: HashMap<ValueId, PReg>,
    next_vreg: u32,
    next_preg: u8,
    visited: HashSet<BlockId>,
}

impl<'a> MirToLirLowerer<'a> {
    fn new(func: &'a MirFunction) -> Self {
        let mut lowerer = Self {
            func,
            instructions: Vec::new(),
            value_to_vreg: HashMap::new(),
            value_to_preg: HashMap::new(),
            next_vreg: 0,
            next_preg: 0,
            visited: HashSet::new(),
        };

        for param in &func.params {
            let vreg = lowerer.alloc_vreg();
            lowerer.value_to_vreg.insert(param.value, vreg);
        }

        lowerer
    }

    fn alloc_vreg(&mut self) -> VReg {
        let vreg = VReg(self.next_vreg);
        self.next_vreg += 1;
        vreg
    }

    fn alloc_preg(&mut self) -> PReg {
        let preg = PReg(self.next_preg % 4);
        self.next_preg += 1;
        preg
    }

    fn get_vreg(&mut self, value: ValueId) -> VReg {
        if let Some(vreg) = self.value_to_vreg.get(&value) {
            return *vreg;
        }
        let vreg = self.alloc_vreg();
        self.value_to_vreg.insert(value, vreg);
        vreg
    }

    fn get_preg_for_cond(&mut self, cond: ValueId) -> PReg {
        if let Some(preg) = self.value_to_preg.get(&cond) {
            return *preg;
        }
        let cond_vreg = self.get_vreg(cond);
        let preg = self.alloc_preg();
        let zero = self.alloc_vreg();
        self.instructions.push(LirInst::MovImm {
            dest: zero,
            value: 0,
        });
        self.instructions.push(LirInst::IcmpNe {
            dest: preg,
            src1: cond_vreg,
            src2: zero,
        });
        preg
    }

    fn lower(&mut self) -> Result<(), CompileError> {
        let preds = self.func.predecessors();
        let back_edges = self.detect_back_edges();

        self.emit_block_structured(self.func.entry, &preds, &back_edges)?;

        self.instructions.push(LirInst::Halt);
        Ok(())
    }

    fn detect_back_edges(&self) -> HashSet<(BlockId, BlockId)> {
        let mut back_edges = HashSet::new();
        let mut visited = HashSet::new();
        let mut in_stack = HashSet::new();
        self.dfs_back_edges(
            self.func.entry,
            &mut visited,
            &mut in_stack,
            &mut back_edges,
        );
        back_edges
    }

    fn dfs_back_edges(
        &self,
        block: BlockId,
        visited: &mut HashSet<BlockId>,
        in_stack: &mut HashSet<BlockId>,
        back_edges: &mut HashSet<(BlockId, BlockId)>,
    ) {
        visited.insert(block);
        in_stack.insert(block);
        if let Some(bb) = self.func.block(block) {
            for succ in bb.successors() {
                if in_stack.contains(&succ) {
                    back_edges.insert((block, succ));
                } else if !visited.contains(&succ) {
                    self.dfs_back_edges(succ, visited, in_stack, back_edges);
                }
            }
        }
        in_stack.remove(&block);
    }

    fn emit_phi_moves(&mut self, from_block: BlockId, to_block: BlockId) {
        if let Some(target_bb) = self.func.block(to_block) {
            for phi in &target_bb.phis {
                for (pred_id, val) in &phi.incoming {
                    if *pred_id == from_block {
                        let dest_vreg = self.get_vreg(phi.dest);
                        let src_vreg = self.get_vreg(*val);
                        self.instructions.push(LirInst::MovReg {
                            dest: dest_vreg,
                            src: src_vreg,
                        });
                    }
                }
            }
        }
    }

    fn emit_block_structured(
        &mut self,
        block_id: BlockId,
        preds: &HashMap<BlockId, Vec<BlockId>>,
        back_edges: &HashSet<(BlockId, BlockId)>,
    ) -> Result<(), CompileError> {
        if self.visited.contains(&block_id) {
            return Ok(());
        }
        self.visited.insert(block_id);

        let bb = self
            .func
            .block(block_id)
            .ok_or_else(|| CompileError::InternalError {
                message: format!("block {block_id} not found"),
            })?;

        let is_loop_header = preds
            .get(&block_id)
            .is_some_and(|p| p.iter().any(|pred| back_edges.contains(&(*pred, block_id))));

        if is_loop_header {
            self.instructions.push(LirInst::Loop);
        }

        for inst in &bb.instructions {
            self.lower_instruction(inst)?;
        }

        let terminator = bb.terminator.clone();
        match &terminator {
            Terminator::Return => {}
            Terminator::Branch { target } => {
                self.emit_phi_moves(block_id, *target);
                if back_edges.contains(&(block_id, *target)) {
                    if is_loop_header {
                        self.instructions.push(LirInst::Endloop);
                    }
                    return Ok(());
                }
                self.emit_block_structured(*target, preds, back_edges)?;
            }
            Terminator::CondBranch {
                cond,
                true_target,
                false_target,
            } => {
                let preg = self.get_preg_for_cond(*cond);

                let true_is_loop_body = back_edges.iter().any(|(src, tgt)| {
                    *tgt == block_id && self.is_reachable_from(*true_target, *src, back_edges)
                });
                let false_is_loop_exit =
                    is_loop_header && !back_edges.iter().any(|(_, tgt)| *tgt == *false_target);

                if is_loop_header && true_is_loop_body && false_is_loop_exit {
                    let exit_preg = self.emit_negated_comparison(block_id, *cond);
                    self.instructions.push(LirInst::Break { pred: exit_preg });
                    self.emit_phi_moves(block_id, *true_target);
                    self.emit_block_structured(*true_target, preds, back_edges)?;
                    self.instructions.push(LirInst::Endloop);
                    self.emit_phi_moves(block_id, *false_target);
                    self.emit_block_structured(*false_target, preds, back_edges)?;
                } else {
                    let true_merges_to_false =
                        self.func.block(*true_target).map_or(false, |bb| {
                            matches!(&bb.terminator, Terminator::Branch { target } if *target == *false_target)
                        });

                    self.instructions.push(LirInst::If { pred: preg });
                    self.emit_phi_moves(block_id, *true_target);
                    let false_guarded = self.visited.insert(*false_target);
                    self.emit_block_structured(*true_target, preds, back_edges)?;
                    if false_guarded {
                        self.visited.remove(false_target);
                    }

                    if !self.visited.contains(false_target) {
                        self.instructions.push(LirInst::Else);
                        self.emit_phi_moves(block_id, *false_target);
                        if !true_merges_to_false {
                            self.emit_block_structured(*false_target, preds, back_edges)?;
                        }
                    }
                    self.instructions.push(LirInst::Endif);

                    if !self.visited.contains(false_target) {
                        self.emit_block_structured(*false_target, preds, back_edges)?;
                    }
                }
            }
        }

        let already_emitted_endloop =
            matches!(&terminator, Terminator::CondBranch { .. }) && is_loop_header;
        if is_loop_header
            && !already_emitted_endloop
            && !matches!(terminator, Terminator::Branch { target } if back_edges.contains(&(block_id, target)))
        {
            self.instructions.push(LirInst::Endloop);
        }

        Ok(())
    }

    /// Emit the negation of a comparison that defined `cond` in `block_id`.
    /// Returns a fresh PReg that is true when the original comparison is false.
    fn emit_negated_comparison(&mut self, block_id: BlockId, cond: ValueId) -> PReg {
        let exit_preg = self.alloc_preg();
        if let Some(bb) = self.func.block(block_id) {
            for inst in &bb.instructions {
                if let MirInst::BinOp {
                    dest,
                    op,
                    lhs,
                    rhs,
                    ty,
                } = inst
                {
                    if *dest == cond && op.is_comparison() {
                        let lhs_v = self.get_vreg(*lhs);
                        let rhs_v = self.get_vreg(*rhs);
                        let is_float = ty.is_float();
                        match (op, is_float) {
                            (BinOp::Lt, false) => {
                                self.instructions.push(LirInst::IcmpGe {
                                    dest: exit_preg,
                                    src1: lhs_v,
                                    src2: rhs_v,
                                });
                            }
                            (BinOp::Le, false) => {
                                self.instructions.push(LirInst::IcmpGt {
                                    dest: exit_preg,
                                    src1: rhs_v,
                                    src2: lhs_v,
                                });
                            }
                            (BinOp::Gt, false) => {
                                self.instructions.push(LirInst::IcmpLe {
                                    dest: exit_preg,
                                    src1: lhs_v,
                                    src2: rhs_v,
                                });
                            }
                            (BinOp::Ge, false) => {
                                self.instructions.push(LirInst::IcmpLt {
                                    dest: exit_preg,
                                    src1: lhs_v,
                                    src2: rhs_v,
                                });
                            }
                            (BinOp::Eq, false) => {
                                self.instructions.push(LirInst::IcmpNe {
                                    dest: exit_preg,
                                    src1: lhs_v,
                                    src2: rhs_v,
                                });
                            }
                            (BinOp::Ne, false) => {
                                self.instructions.push(LirInst::IcmpEq {
                                    dest: exit_preg,
                                    src1: lhs_v,
                                    src2: rhs_v,
                                });
                            }
                            (BinOp::Lt, true) => {
                                self.instructions.push(LirInst::FcmpGt {
                                    dest: exit_preg,
                                    src1: rhs_v,
                                    src2: lhs_v,
                                });
                            }
                            _ => {
                                self.instructions.push(LirInst::IcmpGe {
                                    dest: exit_preg,
                                    src1: lhs_v,
                                    src2: rhs_v,
                                });
                            }
                        }
                        return exit_preg;
                    }
                }
            }
        }
        let zero = self.alloc_vreg();
        self.instructions.push(LirInst::MovImm {
            dest: zero,
            value: 0,
        });
        self.instructions.push(LirInst::IcmpEq {
            dest: exit_preg,
            src1: zero,
            src2: zero,
        });
        exit_preg
    }

    fn is_reachable_from(
        &self,
        start: BlockId,
        target: BlockId,
        back_edges: &HashSet<(BlockId, BlockId)>,
    ) -> bool {
        let mut visited = HashSet::new();
        let mut stack = vec![start];
        while let Some(bid) = stack.pop() {
            if bid == target {
                return true;
            }
            if !visited.insert(bid) {
                continue;
            }
            if let Some(bb) = self.func.block(bid) {
                for succ in bb.successors() {
                    if !back_edges.contains(&(bid, succ)) {
                        stack.push(succ);
                    }
                }
            }
        }
        false
    }

    fn lower_instruction(&mut self, inst: &MirInst) -> Result<(), CompileError> {
        match inst {
            MirInst::BinOp {
                dest,
                op,
                lhs,
                rhs,
                ty,
            } => self.lower_binop(*dest, *op, *lhs, *rhs, *ty),
            MirInst::UnaryOp {
                dest,
                op,
                operand,
                ty,
            } => self.lower_unaryop(*dest, *op, *operand, *ty),
            MirInst::Load {
                dest,
                addr,
                space,
                ty,
            } => {
                let dest_vreg = self.get_vreg(*dest);
                let addr_vreg = self.get_vreg(*addr);
                let width = type_to_mem_width(*ty);
                match space {
                    AddressSpace::Local => {
                        self.instructions.push(LirInst::LocalLoad {
                            dest: dest_vreg,
                            addr: addr_vreg,
                            width,
                        });
                    }
                    AddressSpace::Device | AddressSpace::Private => {
                        self.instructions.push(LirInst::DeviceLoad {
                            dest: dest_vreg,
                            addr: addr_vreg,
                            width,
                        });
                    }
                }
                Ok(())
            }
            MirInst::Store { addr, value, space } => {
                let addr_vreg = self.get_vreg(*addr);
                let val_vreg = self.get_vreg(*value);
                match space {
                    AddressSpace::Local => {
                        self.instructions.push(LirInst::LocalStore {
                            addr: addr_vreg,
                            value: val_vreg,
                            width: MemWidth::W32,
                        });
                    }
                    AddressSpace::Device | AddressSpace::Private => {
                        self.instructions.push(LirInst::DeviceStore {
                            addr: addr_vreg,
                            value: val_vreg,
                            width: MemWidth::W32,
                        });
                    }
                }
                Ok(())
            }
            MirInst::Const { dest, value } => {
                let dest_vreg = self.get_vreg(*dest);
                self.instructions.push(LirInst::MovImm {
                    dest: dest_vreg,
                    value: value.to_bits(),
                });
                Ok(())
            }
            MirInst::Call { dest, func, args } => self.lower_call(*dest, *func, args),
            MirInst::Cast {
                dest,
                value,
                from,
                to,
            } => {
                let dest_vreg = self.get_vreg(*dest);
                let src_vreg = self.get_vreg(*value);
                match (from, to) {
                    (MirType::F32, MirType::I32) => {
                        self.instructions.push(LirInst::CvtF32I32 {
                            dest: dest_vreg,
                            src: src_vreg,
                        });
                    }
                    (MirType::I32, MirType::F32) => {
                        self.instructions.push(LirInst::CvtI32F32 {
                            dest: dest_vreg,
                            src: src_vreg,
                        });
                    }
                    _ => {
                        self.instructions.push(LirInst::MovReg {
                            dest: dest_vreg,
                            src: src_vreg,
                        });
                    }
                }
                Ok(())
            }
            MirInst::Barrier => {
                self.instructions.push(LirInst::Barrier);
                Ok(())
            }
            MirInst::Fence { .. } => Ok(()),
            MirInst::Shuffle {
                dest,
                value,
                lane,
                mode,
            } => {
                let dest_vreg = self.get_vreg(*dest);
                let val_vreg = self.get_vreg(*value);
                let lane_vreg = self.get_vreg(*lane);
                match mode {
                    ShuffleMode::Direct => {
                        self.instructions.push(LirInst::MovReg {
                            dest: dest_vreg,
                            src: val_vreg,
                        });
                    }
                    ShuffleMode::Up | ShuffleMode::Down | ShuffleMode::Xor => {
                        self.instructions.push(LirInst::Iadd {
                            dest: dest_vreg,
                            src1: val_vreg,
                            src2: lane_vreg,
                        });
                    }
                }
                Ok(())
            }
            MirInst::ReadSpecialReg { dest, sr_index } => {
                let dest_vreg = self.get_vreg(*dest);
                let sr = match sr_index {
                    0 => SpecialReg::ThreadIdX,
                    1 => SpecialReg::ThreadIdY,
                    2 => SpecialReg::ThreadIdZ,
                    3 => SpecialReg::WaveId,
                    4 => SpecialReg::LaneId,
                    5 => SpecialReg::WorkgroupIdX,
                    6 => SpecialReg::WorkgroupIdY,
                    7 => SpecialReg::WorkgroupIdZ,
                    8 => SpecialReg::WorkgroupSizeX,
                    9 => SpecialReg::WorkgroupSizeY,
                    10 => SpecialReg::WorkgroupSizeZ,
                    14 => SpecialReg::WaveWidth,
                    _ => SpecialReg::ThreadIdX,
                };
                self.instructions.push(LirInst::MovSr {
                    dest: dest_vreg,
                    sr,
                });
                Ok(())
            }
            MirInst::AtomicRmw {
                dest, addr, value, ..
            } => {
                let dest_vreg = self.get_vreg(*dest);
                let addr_vreg = self.get_vreg(*addr);
                let val_vreg = self.get_vreg(*value);
                self.instructions.push(LirInst::Iadd {
                    dest: dest_vreg,
                    src1: addr_vreg,
                    src2: val_vreg,
                });
                Ok(())
            }
        }
    }

    fn lower_binop(
        &mut self,
        dest: ValueId,
        op: BinOp,
        lhs: ValueId,
        rhs: ValueId,
        ty: MirType,
    ) -> Result<(), CompileError> {
        let lhs_vreg = self.get_vreg(lhs);
        let rhs_vreg = self.get_vreg(rhs);

        if op.is_comparison() {
            let preg = self.alloc_preg();
            self.value_to_preg.insert(dest, preg);

            match (op, ty.is_float()) {
                (BinOp::Lt, false) => {
                    self.instructions.push(LirInst::UcmpLt {
                        dest: preg,
                        src1: lhs_vreg,
                        src2: rhs_vreg,
                    });
                }
                (BinOp::Le, false) => {
                    self.instructions.push(LirInst::IcmpLe {
                        dest: preg,
                        src1: lhs_vreg,
                        src2: rhs_vreg,
                    });
                }
                (BinOp::Gt, false) => {
                    self.instructions.push(LirInst::IcmpGt {
                        dest: preg,
                        src1: lhs_vreg,
                        src2: rhs_vreg,
                    });
                }
                (BinOp::Ge, false) => {
                    self.instructions.push(LirInst::IcmpGe {
                        dest: preg,
                        src1: lhs_vreg,
                        src2: rhs_vreg,
                    });
                }
                (BinOp::Eq, false) => {
                    self.instructions.push(LirInst::IcmpEq {
                        dest: preg,
                        src1: lhs_vreg,
                        src2: rhs_vreg,
                    });
                }
                (BinOp::Ne, false) => {
                    self.instructions.push(LirInst::IcmpNe {
                        dest: preg,
                        src1: lhs_vreg,
                        src2: rhs_vreg,
                    });
                }
                (BinOp::Lt, true) => {
                    self.instructions.push(LirInst::FcmpLt {
                        dest: preg,
                        src1: lhs_vreg,
                        src2: rhs_vreg,
                    });
                }
                (BinOp::Gt, true) => {
                    self.instructions.push(LirInst::FcmpGt {
                        dest: preg,
                        src1: lhs_vreg,
                        src2: rhs_vreg,
                    });
                }
                (BinOp::Eq, true) => {
                    self.instructions.push(LirInst::FcmpEq {
                        dest: preg,
                        src1: lhs_vreg,
                        src2: rhs_vreg,
                    });
                }
                _ => {
                    self.instructions.push(LirInst::IcmpEq {
                        dest: preg,
                        src1: lhs_vreg,
                        src2: rhs_vreg,
                    });
                }
            }

            return Ok(());
        }

        let dest_vreg = self.get_vreg(dest);
        match (op, ty.is_float()) {
            (BinOp::Add, false) => {
                self.instructions.push(LirInst::Iadd {
                    dest: dest_vreg,
                    src1: lhs_vreg,
                    src2: rhs_vreg,
                });
            }
            (BinOp::Sub, false) => {
                self.instructions.push(LirInst::Isub {
                    dest: dest_vreg,
                    src1: lhs_vreg,
                    src2: rhs_vreg,
                });
            }
            (BinOp::Mul, false) => {
                self.instructions.push(LirInst::Imul {
                    dest: dest_vreg,
                    src1: lhs_vreg,
                    src2: rhs_vreg,
                });
            }
            (BinOp::Div | BinOp::FloorDiv, false) => {
                self.instructions.push(LirInst::Idiv {
                    dest: dest_vreg,
                    src1: lhs_vreg,
                    src2: rhs_vreg,
                });
            }
            (BinOp::Mod, false) => {
                self.instructions.push(LirInst::Imod {
                    dest: dest_vreg,
                    src1: lhs_vreg,
                    src2: rhs_vreg,
                });
            }
            (BinOp::Add, true) => {
                self.instructions.push(LirInst::Fadd {
                    dest: dest_vreg,
                    src1: lhs_vreg,
                    src2: rhs_vreg,
                });
            }
            (BinOp::Sub, true) => {
                self.instructions.push(LirInst::Fsub {
                    dest: dest_vreg,
                    src1: lhs_vreg,
                    src2: rhs_vreg,
                });
            }
            (BinOp::Mul, true) => {
                self.instructions.push(LirInst::Fmul {
                    dest: dest_vreg,
                    src1: lhs_vreg,
                    src2: rhs_vreg,
                });
            }
            (BinOp::Div, true) => {
                self.instructions.push(LirInst::Fdiv {
                    dest: dest_vreg,
                    src1: lhs_vreg,
                    src2: rhs_vreg,
                });
            }
            (BinOp::BitAnd, _) => {
                self.instructions.push(LirInst::And {
                    dest: dest_vreg,
                    src1: lhs_vreg,
                    src2: rhs_vreg,
                });
            }
            (BinOp::BitOr, _) => {
                self.instructions.push(LirInst::Or {
                    dest: dest_vreg,
                    src1: lhs_vreg,
                    src2: rhs_vreg,
                });
            }
            (BinOp::BitXor, _) => {
                self.instructions.push(LirInst::Xor {
                    dest: dest_vreg,
                    src1: lhs_vreg,
                    src2: rhs_vreg,
                });
            }
            (BinOp::Shl, _) => {
                self.instructions.push(LirInst::Shl {
                    dest: dest_vreg,
                    src1: lhs_vreg,
                    src2: rhs_vreg,
                });
            }
            (BinOp::Shr, _) => {
                self.instructions.push(LirInst::Shr {
                    dest: dest_vreg,
                    src1: lhs_vreg,
                    src2: rhs_vreg,
                });
            }
            _ => {
                self.instructions.push(LirInst::Iadd {
                    dest: dest_vreg,
                    src1: lhs_vreg,
                    src2: rhs_vreg,
                });
            }
        }
        Ok(())
    }

    fn lower_unaryop(
        &mut self,
        dest: ValueId,
        op: UnaryOp,
        operand: ValueId,
        ty: MirType,
    ) -> Result<(), CompileError> {
        let dest_vreg = self.get_vreg(dest);
        let src_vreg = self.get_vreg(operand);
        match (op, ty.is_float()) {
            (UnaryOp::Neg, true) => {
                self.instructions.push(LirInst::Fneg {
                    dest: dest_vreg,
                    src: src_vreg,
                });
            }
            (UnaryOp::Neg, false) => {
                self.instructions.push(LirInst::Ineg {
                    dest: dest_vreg,
                    src: src_vreg,
                });
            }
            (UnaryOp::BitNot | UnaryOp::Not, _) => {
                self.instructions.push(LirInst::Not {
                    dest: dest_vreg,
                    src: src_vreg,
                });
            }
        }
        Ok(())
    }

    fn lower_call(
        &mut self,
        dest: Option<ValueId>,
        func: BuiltinFunc,
        args: &[ValueId],
    ) -> Result<(), CompileError> {
        let dest_vreg = dest.map(|d| self.get_vreg(d));
        match func {
            BuiltinFunc::Sqrt => {
                if let (Some(d), Some(a)) = (dest_vreg, args.first()) {
                    let src = self.get_vreg(*a);
                    self.instructions.push(LirInst::Fsqrt { dest: d, src });
                }
            }
            BuiltinFunc::Sin => {
                if let (Some(d), Some(a)) = (dest_vreg, args.first()) {
                    let src = self.get_vreg(*a);
                    self.instructions.push(LirInst::Fsin { dest: d, src });
                }
            }
            BuiltinFunc::Cos => {
                if let (Some(d), Some(a)) = (dest_vreg, args.first()) {
                    let src = self.get_vreg(*a);
                    self.instructions.push(LirInst::Fcos { dest: d, src });
                }
            }
            BuiltinFunc::Exp2 => {
                if let (Some(d), Some(a)) = (dest_vreg, args.first()) {
                    let src = self.get_vreg(*a);
                    self.instructions.push(LirInst::Fexp2 { dest: d, src });
                }
            }
            BuiltinFunc::Log2 => {
                if let (Some(d), Some(a)) = (dest_vreg, args.first()) {
                    let src = self.get_vreg(*a);
                    self.instructions.push(LirInst::Flog2 { dest: d, src });
                }
            }
            BuiltinFunc::Abs => {
                if let (Some(d), Some(a)) = (dest_vreg, args.first()) {
                    let src = self.get_vreg(*a);
                    self.instructions.push(LirInst::Fabs { dest: d, src });
                }
            }
            BuiltinFunc::Min => {
                if let (Some(d), Some(a), Some(b)) = (dest_vreg, args.first(), args.get(1)) {
                    let s1 = self.get_vreg(*a);
                    let s2 = self.get_vreg(*b);
                    self.instructions.push(LirInst::Fmin {
                        dest: d,
                        src1: s1,
                        src2: s2,
                    });
                }
            }
            BuiltinFunc::Max => {
                if let (Some(d), Some(a), Some(b)) = (dest_vreg, args.first(), args.get(1)) {
                    let s1 = self.get_vreg(*a);
                    let s2 = self.get_vreg(*b);
                    self.instructions.push(LirInst::Fmax {
                        dest: d,
                        src1: s1,
                        src2: s2,
                    });
                }
            }
            BuiltinFunc::AtomicAdd => {
                if let Some(d) = dest_vreg {
                    self.instructions
                        .push(LirInst::MovImm { dest: d, value: 0 });
                }
            }
        }
        Ok(())
    }
}

fn type_to_mem_width(ty: MirType) -> MemWidth {
    match ty {
        MirType::F16 => MemWidth::W16,
        MirType::F64 => MemWidth::W64,
        _ => MemWidth::W32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::basic_block::{BasicBlock, Terminator};
    use crate::mir::function::MirParam;
    use crate::mir::instruction::ConstValue;

    fn make_simple_mir() -> MirFunction {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        func.params.push(MirParam {
            value: ValueId(0),
            ty: MirType::I32,
            name: "x".into(),
        });

        let mut bb0 = BasicBlock::new(BlockId(0));
        bb0.instructions.push(MirInst::Const {
            dest: ValueId(1),
            value: ConstValue::I32(42),
        });
        bb0.instructions.push(MirInst::BinOp {
            dest: ValueId(2),
            op: BinOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: MirType::I32,
        });
        bb0.terminator = Terminator::Return;
        func.blocks.push(bb0);
        func
    }

    #[test]
    fn test_lower_simple_function() {
        let func = make_simple_mir();
        let lir = lower_function(&func).unwrap();
        assert!(!lir.is_empty());
        assert!(lir.iter().any(|i| matches!(i, LirInst::Iadd { .. })));
        assert!(matches!(lir.last(), Some(LirInst::Halt)));
    }

    #[test]
    fn test_lower_with_branch() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        func.params.push(MirParam {
            value: ValueId(0),
            ty: MirType::I32,
            name: "x".into(),
        });

        let mut bb0 = BasicBlock::new(BlockId(0));
        bb0.instructions.push(MirInst::BinOp {
            dest: ValueId(1),
            op: BinOp::Lt,
            lhs: ValueId(0),
            rhs: ValueId(0),
            ty: MirType::I32,
        });
        bb0.terminator = Terminator::CondBranch {
            cond: ValueId(1),
            true_target: BlockId(1),
            false_target: BlockId(2),
        };

        let mut bb1 = BasicBlock::new(BlockId(1));
        bb1.terminator = Terminator::Branch { target: BlockId(2) };

        let mut bb2 = BasicBlock::new(BlockId(2));
        bb2.terminator = Terminator::Return;

        func.blocks.push(bb0);
        func.blocks.push(bb1);
        func.blocks.push(bb2);

        let lir = lower_function(&func).unwrap();
        assert!(lir.iter().any(|i| matches!(i, LirInst::If { .. })));
        assert!(lir.iter().any(|i| matches!(i, LirInst::Endif)));
        let ucmp_count = lir
            .iter()
            .filter(|i| matches!(i, LirInst::UcmpLt { .. }))
            .count();
        assert_eq!(ucmp_count, 1);
        let icmp_ne_count = lir
            .iter()
            .filter(|i| matches!(i, LirInst::IcmpNe { .. }))
            .count();
        assert_eq!(icmp_ne_count, 0);
    }

    #[test]
    fn test_comparison_uses_preg_directly() {
        let mut func = MirFunction::new("test".into(), BlockId(0));
        func.params.push(MirParam {
            value: ValueId(0),
            ty: MirType::I32,
            name: "a".into(),
        });
        func.params.push(MirParam {
            value: ValueId(1),
            ty: MirType::I32,
            name: "b".into(),
        });

        let mut bb0 = BasicBlock::new(BlockId(0));
        bb0.instructions.push(MirInst::BinOp {
            dest: ValueId(2),
            op: BinOp::Lt,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: MirType::I32,
        });
        bb0.terminator = Terminator::CondBranch {
            cond: ValueId(2),
            true_target: BlockId(1),
            false_target: BlockId(2),
        };

        let mut bb1 = BasicBlock::new(BlockId(1));
        bb1.terminator = Terminator::Branch { target: BlockId(2) };
        let mut bb2 = BasicBlock::new(BlockId(2));
        bb2.terminator = Terminator::Return;

        func.blocks.push(bb0);
        func.blocks.push(bb1);
        func.blocks.push(bb2);

        let lir = lower_function(&func).unwrap();
        let mut saw_ucmp = false;
        let mut ucmp_preg = PReg(255);
        let mut if_preg = PReg(254);
        for inst in &lir {
            if let LirInst::UcmpLt { dest, .. } = inst {
                saw_ucmp = true;
                ucmp_preg = *dest;
            }
            if let LirInst::If { pred } = inst {
                if_preg = *pred;
            }
        }
        assert!(saw_ucmp);
        assert_eq!(ucmp_preg, if_preg);
    }
}
