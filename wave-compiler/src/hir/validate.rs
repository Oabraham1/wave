// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! HIR validation and type checking for WAVE GPU kernels.
//!
//! Validates kernel definitions by checking that all variables are defined
//! before use, types are consistent, and control flow is well-formed.

use std::collections::HashMap;

use super::expr::{BuiltinFunc, Expr, Literal, UnaryOp};
use super::kernel::Kernel;
use super::stmt::Stmt;
use super::types::Type;
use crate::diagnostics::error::CompileError;

/// Type environment mapping variable names to their types.
struct TypeEnv {
    scopes: Vec<HashMap<String, Type>>,
}

impl TypeEnv {
    fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
        }
    }

    fn define(&mut self, name: &str, ty: Type) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), ty);
        }
    }

    fn lookup(&self, name: &str) -> Option<&Type> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
            }
        }
        None
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }
}

/// Validate and type-check a kernel definition.
///
/// # Errors
///
/// Returns `CompileError` if the kernel has type errors or undefined variables.
pub fn validate_kernel(kernel: &Kernel) -> Result<(), CompileError> {
    let mut env = TypeEnv::new();

    for param in &kernel.params {
        env.define(&param.name, param.ty.clone());
    }

    validate_stmts(&kernel.body, &mut env)
}

fn validate_stmts(stmts: &[Stmt], env: &mut TypeEnv) -> Result<(), CompileError> {
    for stmt in stmts {
        validate_stmt(stmt, env)?;
    }
    Ok(())
}

fn validate_stmt(stmt: &Stmt, env: &mut TypeEnv) -> Result<(), CompileError> {
    match stmt {
        Stmt::Assign { target, value } => {
            let ty = infer_type(value, env)?;
            env.define(target, ty);
            Ok(())
        }
        Stmt::If {
            condition,
            then_body,
            else_body,
        } => {
            let cond_ty = infer_type(condition, env)?;
            if cond_ty != Type::Bool {
                return Err(CompileError::TypeMismatch {
                    expected: "bool".into(),
                    found: format!("{cond_ty}"),
                });
            }
            env.push_scope();
            validate_stmts(then_body, env)?;
            env.pop_scope();
            if let Some(else_stmts) = else_body {
                env.push_scope();
                validate_stmts(else_stmts, env)?;
                env.pop_scope();
            }
            Ok(())
        }
        Stmt::For {
            var,
            start,
            end,
            step,
            body,
        } => {
            infer_type(start, env)?;
            infer_type(end, env)?;
            infer_type(step, env)?;
            env.push_scope();
            env.define(var, Type::I32);
            validate_stmts(body, env)?;
            env.pop_scope();
            Ok(())
        }
        Stmt::While { condition, body } => {
            let cond_ty = infer_type(condition, env)?;
            if cond_ty != Type::Bool {
                return Err(CompileError::TypeMismatch {
                    expected: "bool".into(),
                    found: format!("{cond_ty}"),
                });
            }
            env.push_scope();
            validate_stmts(body, env)?;
            env.pop_scope();
            Ok(())
        }
        Stmt::Return { value } => {
            if let Some(val) = value {
                infer_type(val, env)?;
            }
            Ok(())
        }
        Stmt::Store { addr, value, .. } => {
            infer_type(addr, env)?;
            infer_type(value, env)?;
            Ok(())
        }
        Stmt::Barrier | Stmt::Fence { .. } => Ok(()),
    }
}

/// Infer the type of an expression given a type environment.
///
/// # Errors
///
/// Returns `CompileError` if a variable is undefined or types are incompatible.
fn infer_type(expr: &Expr, env: &TypeEnv) -> Result<Type, CompileError> {
    match expr {
        Expr::Var(name) => env
            .lookup(name)
            .cloned()
            .ok_or_else(|| CompileError::UndefinedVariable { name: name.clone() }),
        Expr::Literal(lit) => Ok(match lit {
            Literal::Int(_) => Type::I32,
            Literal::UInt(_) => Type::U32,
            Literal::Float(_) => Type::F32,
            Literal::Bool(_) => Type::Bool,
        }),
        Expr::BinOp { op, lhs, .. } => {
            let lhs_ty = infer_type(lhs, env)?;
            if op.is_comparison() {
                Ok(Type::Bool)
            } else {
                Ok(lhs_ty)
            }
        }
        Expr::UnaryOp { op, operand } => {
            let operand_ty = infer_type(operand, env)?;
            match op {
                UnaryOp::Not => Ok(Type::Bool),
                UnaryOp::Neg | UnaryOp::BitNot => Ok(operand_ty),
            }
        }
        Expr::Call { func, .. } => Ok(match func {
            BuiltinFunc::Sqrt
            | BuiltinFunc::Sin
            | BuiltinFunc::Cos
            | BuiltinFunc::Exp2
            | BuiltinFunc::Log2 => Type::F32,
            BuiltinFunc::Abs | BuiltinFunc::Min | BuiltinFunc::Max | BuiltinFunc::AtomicAdd => {
                Type::U32
            }
        }),
        Expr::Index { base, .. } => {
            let base_ty = infer_type(base, env)?;
            match base_ty {
                Type::Ptr(_) => Ok(Type::F32),
                Type::Array(elem, _) => Ok(*elem),
                _ => Ok(Type::F32),
            }
        }
        Expr::Cast { to, .. } => Ok(to.clone()),
        Expr::ThreadId(_)
        | Expr::WorkgroupId(_)
        | Expr::WorkgroupSize(_)
        | Expr::LaneId
        | Expr::WaveWidth => Ok(Type::U32),
        Expr::Load { .. } => Ok(Type::F32),
        Expr::Shuffle { .. } => Ok(Type::U32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::expr::{BinOp, Dimension};
    use crate::hir::kernel::{KernelAttributes, KernelParam};
    use crate::hir::types::AddressSpace;

    #[test]
    fn test_validate_simple_kernel() {
        let kernel = Kernel {
            name: "test".into(),
            params: vec![KernelParam {
                name: "n".into(),
                ty: Type::U32,
                address_space: AddressSpace::Private,
            }],
            body: vec![
                Stmt::Assign {
                    target: "gid".into(),
                    value: Expr::ThreadId(Dimension::X),
                },
                Stmt::If {
                    condition: Expr::BinOp {
                        op: BinOp::Lt,
                        lhs: Box::new(Expr::Var("gid".into())),
                        rhs: Box::new(Expr::Var("n".into())),
                    },
                    then_body: vec![Stmt::Assign {
                        target: "x".into(),
                        value: Expr::Literal(Literal::Int(1)),
                    }],
                    else_body: None,
                },
            ],
            attributes: KernelAttributes::default(),
        };
        assert!(validate_kernel(&kernel).is_ok());
    }

    #[test]
    fn test_validate_undefined_variable() {
        let kernel = Kernel {
            name: "test".into(),
            params: vec![],
            body: vec![Stmt::Assign {
                target: "x".into(),
                value: Expr::Var("undefined_var".into()),
            }],
            attributes: KernelAttributes::default(),
        };
        assert!(validate_kernel(&kernel).is_err());
    }

    #[test]
    fn test_infer_literal_types() {
        let env = TypeEnv::new();
        assert_eq!(
            infer_type(&Expr::Literal(Literal::Int(42)), &env).unwrap(),
            Type::I32
        );
        assert_eq!(
            infer_type(&Expr::Literal(Literal::Float(1.0)), &env).unwrap(),
            Type::F32
        );
        assert_eq!(
            infer_type(&Expr::Literal(Literal::Bool(true)), &env).unwrap(),
            Type::Bool
        );
    }

    #[test]
    fn test_infer_thread_id_type() {
        let env = TypeEnv::new();
        assert_eq!(
            infer_type(&Expr::ThreadId(Dimension::X), &env).unwrap(),
            Type::U32
        );
    }
}
