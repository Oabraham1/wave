// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! HIR to MIR lowering: flattens structured control flow into a CFG
//! and constructs SSA form with explicit basic blocks and terminators.
//!
//! Each HIR statement is translated into MIR instructions within basic
//! blocks. Control flow constructs (if/else, for, while) are lowered
//! into conditional branches between blocks. For-loop variables use
//! phi nodes at the loop header to merge initial and updated values.

use std::collections::HashMap;

use crate::diagnostics::CompileError;
use crate::hir::expr::{BinOp, BuiltinFunc, Dimension, Expr, Literal};
use crate::hir::kernel::Kernel;
use crate::hir::stmt::Stmt;
use crate::hir::types::AddressSpace;
use crate::mir::basic_block::{PhiNode, Terminator};
use crate::mir::builder::MirBuilder;
use crate::mir::function::MirFunction;
use crate::mir::instruction::{ConstValue, MirInst};
use crate::mir::types::{lower_type, MirType};
use crate::mir::value::ValueId;

/// Lower an HIR kernel to a MIR function.
///
/// # Errors
///
/// Returns `CompileError` if the kernel contains unsupported constructs.
pub fn lower_kernel(kernel: &Kernel) -> Result<MirFunction, CompileError> {
    let mut lowerer = HirToMirLowerer::new(&kernel.name);

    for param in &kernel.params {
        let ty = lower_type(&param.ty);
        let val = lowerer.builder.add_param(param.name.clone(), ty);
        lowerer.variables.insert(param.name.clone(), val);
    }

    let entry = lowerer.builder.current_block();
    lowerer.builder.switch_to_block(entry);

    lowerer.lower_stmts(&kernel.body)?;

    lowerer.builder.set_terminator(Terminator::Return);

    Ok(lowerer.builder.finish())
}

struct HirToMirLowerer {
    builder: MirBuilder,
    variables: HashMap<String, ValueId>,
    variable_types: HashMap<String, MirType>,
}

impl HirToMirLowerer {
    fn new(name: &str) -> Self {
        Self {
            builder: MirBuilder::new(name.to_string()),
            variables: HashMap::new(),
            variable_types: HashMap::new(),
        }
    }

    fn lower_stmts(&mut self, stmts: &[Stmt]) -> Result<(), CompileError> {
        for stmt in stmts {
            self.lower_stmt(stmt)?;
        }
        Ok(())
    }

    fn lower_stmt(&mut self, stmt: &Stmt) -> Result<(), CompileError> {
        match stmt {
            Stmt::Assign { target, value } => {
                let val = self.lower_expr(value)?;
                let ty = self.infer_mir_type(value);
                self.variables.insert(target.clone(), val);
                self.variable_types.insert(target.clone(), ty);
                Ok(())
            }
            Stmt::If {
                condition,
                then_body,
                else_body,
            } => {
                let cond = self.lower_expr(condition)?;
                let then_block = self.builder.create_block();
                let merge_block = self.builder.create_block();

                if let Some(else_stmts) = else_body {
                    let else_block = self.builder.create_block();
                    self.builder.set_terminator(Terminator::CondBranch {
                        cond,
                        true_target: then_block,
                        false_target: else_block,
                    });

                    self.builder.switch_to_block(then_block);
                    self.lower_stmts(then_body)?;
                    self.builder.set_terminator(Terminator::Branch {
                        target: merge_block,
                    });

                    self.builder.switch_to_block(else_block);
                    self.lower_stmts(else_stmts)?;
                    self.builder.set_terminator(Terminator::Branch {
                        target: merge_block,
                    });
                } else {
                    self.builder.set_terminator(Terminator::CondBranch {
                        cond,
                        true_target: then_block,
                        false_target: merge_block,
                    });

                    self.builder.switch_to_block(then_block);
                    self.lower_stmts(then_body)?;
                    self.builder.set_terminator(Terminator::Branch {
                        target: merge_block,
                    });
                }

                self.builder.switch_to_block(merge_block);
                Ok(())
            }
            Stmt::For {
                var,
                start,
                end,
                step,
                body,
            } => {
                let start_val = self.lower_expr(start)?;
                let end_val = self.lower_expr(end)?;
                let step_val = self.lower_expr(step)?;

                let preheader_block = self.builder.current_block();

                let header_block = self.builder.create_block();
                let body_block = self.builder.create_block();
                let latch_block = self.builder.create_block();
                let exit_block = self.builder.create_block();

                self.builder.set_terminator(Terminator::Branch {
                    target: header_block,
                });

                self.builder.switch_to_block(header_block);
                let phi_dest = self.builder.next_value();
                self.builder.emit(MirInst::Const {
                    dest: phi_dest,
                    value: ConstValue::I32(0),
                });

                self.variables.insert(var.clone(), phi_dest);
                self.variable_types.insert(var.clone(), MirType::I32);

                // Create phi placeholders for all other variables in scope.
                // Variables modified in the loop body (e.g. accumulators) need
                // SSA phi nodes at the loop header for loop-carried values.
                let var_entries: Vec<(String, ValueId, MirType)> = self
                    .variables
                    .iter()
                    .filter(|(name, _)| name != &var)
                    .map(|(name, &pre_val)| {
                        let ty = self
                            .variable_types
                            .get(name)
                            .copied()
                            .unwrap_or(MirType::I32);
                        (name.clone(), pre_val, ty)
                    })
                    .collect();
                let loop_carried: Vec<(String, ValueId, ValueId, MirType)> = var_entries
                    .into_iter()
                    .map(|(name, pre_val, ty)| {
                        let phi = self.builder.next_value();
                        (name, pre_val, phi, ty)
                    })
                    .collect();
                for (name, _, phi, _) in &loop_carried {
                    self.variables.insert(name.clone(), *phi);
                }

                let cond = self.builder.emit_binop(BinOp::Lt, phi_dest, end_val, MirType::I32);
                self.builder.set_terminator(Terminator::CondBranch {
                    cond,
                    true_target: body_block,
                    false_target: exit_block,
                });

                self.builder.switch_to_block(body_block);
                self.lower_stmts(body)?;
                self.builder.set_terminator(Terminator::Branch {
                    target: latch_block,
                });

                self.builder.switch_to_block(latch_block);
                let current_var = *self.variables.get(var).ok_or_else(|| {
                    CompileError::InternalError {
                        message: "loop variable missing after body".into(),
                    }
                })?;
                let updated = self.builder.emit_binop(
                    BinOp::Add,
                    current_var,
                    step_val,
                    MirType::I32,
                );
                self.variables.insert(var.clone(), updated);
                self.builder.set_terminator(Terminator::Branch {
                    target: header_block,
                });

                if let Some(header_bb) = self.builder.get_block_mut(header_block) {
                    header_bb.phis.push(PhiNode {
                        dest: phi_dest,
                        ty: MirType::I32,
                        incoming: vec![
                            (preheader_block, start_val),
                            (latch_block, updated),
                        ],
                    });
                    header_bb.instructions.retain(|inst| {
                        !matches!(inst, MirInst::Const { dest, .. } if *dest == phi_dest)
                    });

                    // Add phis for loop-carried variables
                    for (name, pre_val, phi, ty) in &loop_carried {
                        let body_val =
                            self.variables.get(name).copied().unwrap_or(*phi);
                        header_bb.phis.push(PhiNode {
                            dest: *phi,
                            ty: *ty,
                            incoming: vec![
                                (preheader_block, *pre_val),
                                (latch_block, body_val),
                            ],
                        });
                    }
                }

                // Restore loop-carried variables to phi dests for exit block
                for (name, _, phi, _) in &loop_carried {
                    self.variables.insert(name.clone(), *phi);
                }

                self.builder.switch_to_block(exit_block);
                Ok(())
            }
            Stmt::While { condition, body } => {
                let header_block = self.builder.create_block();
                let body_block = self.builder.create_block();
                let exit_block = self.builder.create_block();

                self.builder.set_terminator(Terminator::Branch {
                    target: header_block,
                });

                self.builder.switch_to_block(header_block);
                let cond = self.lower_expr(condition)?;
                self.builder.set_terminator(Terminator::CondBranch {
                    cond,
                    true_target: body_block,
                    false_target: exit_block,
                });

                self.builder.switch_to_block(body_block);
                self.lower_stmts(body)?;
                self.builder.set_terminator(Terminator::Branch {
                    target: header_block,
                });

                self.builder.switch_to_block(exit_block);
                Ok(())
            }
            Stmt::Return { .. } => {
                self.builder.set_terminator(Terminator::Return);
                Ok(())
            }
            Stmt::Store { addr, value, space } => {
                let addr_val = self.lower_expr(addr)?;
                let val = self.lower_expr(value)?;
                self.builder.emit_store(addr_val, val, *space);
                Ok(())
            }
            Stmt::Barrier => {
                self.builder.emit(MirInst::Barrier);
                Ok(())
            }
            Stmt::Fence { scope } => {
                self.builder.emit(MirInst::Fence { scope: *scope });
                Ok(())
            }
        }
    }

    fn lower_expr(&mut self, expr: &Expr) -> Result<ValueId, CompileError> {
        match expr {
            Expr::Var(name) => self.variables.get(name).copied().ok_or_else(|| {
                CompileError::UndefinedVariable {
                    name: name.clone(),
                }
            }),
            Expr::Literal(lit) => match lit {
                Literal::Int(v) => Ok(self.builder.emit_const(ConstValue::I32(*v as i32))),
                Literal::UInt(v) => Ok(self.builder.emit_const(ConstValue::U32(*v as u32))),
                Literal::Float(v) => Ok(self.builder.emit_const(ConstValue::F32(*v as f32))),
                Literal::Bool(v) => Ok(self.builder.emit_const(ConstValue::Bool(*v))),
            },
            Expr::BinOp { op, lhs, rhs } => {
                let l = self.lower_expr(lhs)?;
                let r = self.lower_expr(rhs)?;
                let ty = self.infer_mir_type(lhs);
                Ok(self.builder.emit_binop(*op, l, r, ty))
            }
            Expr::UnaryOp { op, operand } => {
                let val = self.lower_expr(operand)?;
                let ty = self.infer_mir_type(operand);
                let dest = self.builder.next_value();
                self.builder.emit(MirInst::UnaryOp {
                    dest,
                    op: *op,
                    operand: val,
                    ty,
                });
                Ok(dest)
            }
            Expr::Call { func, args } => {
                let arg_vals: Vec<ValueId> = args
                    .iter()
                    .map(|a| self.lower_expr(a))
                    .collect::<Result<_, _>>()?;
                let dest = self.builder.next_value();
                self.builder.emit(MirInst::Call {
                    dest: Some(dest),
                    func: *func,
                    args: arg_vals,
                });
                Ok(dest)
            }
            Expr::Index { base, index } => {
                let base_val = self.lower_expr(base)?;
                let idx_val = self.lower_expr(index)?;
                let elem_size = self.builder.emit_const(ConstValue::U32(4));
                let offset = self.builder.emit_binop(BinOp::Mul, idx_val, elem_size, MirType::I32);
                let addr = self.builder.emit_binop(BinOp::Add, base_val, offset, MirType::I32);
                Ok(self
                    .builder
                    .emit_load(addr, AddressSpace::Device, MirType::F32))
            }
            Expr::Cast { expr, to } => {
                let val = self.lower_expr(expr)?;
                let from = self.infer_mir_type(expr);
                let to_mir = lower_type(to);
                let dest = self.builder.next_value();
                self.builder.emit(MirInst::Cast {
                    dest,
                    value: val,
                    from,
                    to: to_mir,
                });
                Ok(dest)
            }
            Expr::ThreadId(dim) => self.emit_special_reg_read(dim_to_thread_id_sr(*dim)),
            Expr::WorkgroupId(dim) => self.emit_special_reg_read(dim_to_workgroup_id_sr(*dim)),
            Expr::WorkgroupSize(dim) => self.emit_special_reg_read(dim_to_workgroup_size_sr(*dim)),
            Expr::LaneId => self.emit_special_reg_read(4),
            Expr::WaveWidth => self.emit_special_reg_read(14),
            Expr::Load { addr, space } => {
                let addr_val = self.lower_expr(addr)?;
                Ok(self.builder.emit_load(addr_val, *space, MirType::F32))
            }
            Expr::Shuffle { value, lane, mode } => {
                let val = self.lower_expr(value)?;
                let lane_val = self.lower_expr(lane)?;
                let dest = self.builder.next_value();
                self.builder.emit(MirInst::Shuffle {
                    dest,
                    value: val,
                    lane: lane_val,
                    mode: *mode,
                });
                Ok(dest)
            }
        }
    }

    fn emit_special_reg_read(&mut self, sr_index: u8) -> Result<ValueId, CompileError> {
        let dest = self.builder.next_value();
        self.builder.emit(MirInst::ReadSpecialReg { dest, sr_index });
        Ok(dest)
    }

    fn infer_mir_type(&self, expr: &Expr) -> MirType {
        match expr {
            Expr::Literal(Literal::Float(_)) => MirType::F32,
            Expr::Literal(Literal::Bool(_)) => MirType::Bool,
            Expr::Var(name) => {
                self.variable_types.get(name).copied().unwrap_or(MirType::I32)
            }
            Expr::BinOp { op, lhs, .. } => {
                if op.is_comparison() {
                    MirType::Bool
                } else {
                    self.infer_mir_type(lhs)
                }
            }
            Expr::Call { func: BuiltinFunc::Sqrt | BuiltinFunc::Sin | BuiltinFunc::Cos | BuiltinFunc::Exp2 | BuiltinFunc::Log2, .. } => MirType::F32,
            Expr::Call { .. } => MirType::I32,
            Expr::Index { .. } | Expr::Load { .. } => MirType::F32,
            _ => MirType::I32,
        }
    }
}

fn dim_to_thread_id_sr(dim: Dimension) -> u8 {
    match dim {
        Dimension::X => 0,
        Dimension::Y => 1,
        Dimension::Z => 2,
    }
}

fn dim_to_workgroup_id_sr(dim: Dimension) -> u8 {
    match dim {
        Dimension::X => 5,
        Dimension::Y => 6,
        Dimension::Z => 7,
    }
}

fn dim_to_workgroup_size_sr(dim: Dimension) -> u8 {
    match dim {
        Dimension::X => 8,
        Dimension::Y => 9,
        Dimension::Z => 10,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::kernel::{KernelAttributes, KernelParam};
    use crate::hir::types::Type;

    #[test]
    fn test_lower_vector_add() {
        let kernel = Kernel {
            name: "vector_add".into(),
            params: vec![
                KernelParam { name: "a".into(), ty: Type::Ptr(AddressSpace::Device), address_space: AddressSpace::Device },
                KernelParam { name: "b".into(), ty: Type::Ptr(AddressSpace::Device), address_space: AddressSpace::Device },
                KernelParam { name: "out".into(), ty: Type::Ptr(AddressSpace::Device), address_space: AddressSpace::Device },
                KernelParam { name: "n".into(), ty: Type::U32, address_space: AddressSpace::Private },
            ],
            body: vec![
                Stmt::Assign { target: "gid".into(), value: Expr::ThreadId(Dimension::X) },
                Stmt::If {
                    condition: Expr::BinOp {
                        op: BinOp::Lt,
                        lhs: Box::new(Expr::Var("gid".into())),
                        rhs: Box::new(Expr::Var("n".into())),
                    },
                    then_body: vec![Stmt::Assign {
                        target: "result".into(),
                        value: Expr::BinOp {
                            op: BinOp::Add,
                            lhs: Box::new(Expr::Index { base: Box::new(Expr::Var("a".into())), index: Box::new(Expr::Var("gid".into())) }),
                            rhs: Box::new(Expr::Index { base: Box::new(Expr::Var("b".into())), index: Box::new(Expr::Var("gid".into())) }),
                        },
                    }],
                    else_body: None,
                },
            ],
            attributes: KernelAttributes::default(),
        };
        let func = lower_kernel(&kernel).unwrap();
        assert_eq!(func.name, "vector_add");
        assert_eq!(func.params.len(), 4);
        assert!(func.block_count() >= 3);
    }

    #[test]
    fn test_lower_simple_assign() {
        let kernel = Kernel {
            name: "test".into(),
            params: vec![],
            body: vec![Stmt::Assign {
                target: "x".into(),
                value: Expr::Literal(Literal::Int(42)),
            }],
            attributes: KernelAttributes::default(),
        };
        let func = lower_kernel(&kernel).unwrap();
        assert_eq!(func.block_count(), 1);
        assert!(!func.blocks[0].instructions.is_empty());
    }

    #[test]
    fn test_lower_for_loop_has_phi() {
        let kernel = Kernel {
            name: "loop_test".into(),
            params: vec![
                KernelParam { name: "n".into(), ty: Type::U32, address_space: AddressSpace::Private },
            ],
            body: vec![Stmt::For {
                var: "i".into(),
                start: Expr::Literal(Literal::Int(0)),
                end: Expr::Var("n".into()),
                step: Expr::Literal(Literal::Int(1)),
                body: vec![Stmt::Barrier],
            }],
            attributes: KernelAttributes::default(),
        };
        let func = lower_kernel(&kernel).unwrap();
        let header = func.blocks.iter().find(|b| !b.phis.is_empty());
        assert!(header.is_some());
        let phi = &header.unwrap().phis[0];
        assert_eq!(phi.incoming.len(), 2);
    }

    #[test]
    fn test_infer_float_type() {
        let lowerer = HirToMirLowerer::new("test");
        assert_eq!(lowerer.infer_mir_type(&Expr::Literal(Literal::Float(1.0))), MirType::F32);
        assert_eq!(lowerer.infer_mir_type(&Expr::Literal(Literal::Int(1))), MirType::I32);
    }
}
