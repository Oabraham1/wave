// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! HIR statement types for WAVE GPU kernels.
//!
//! Statements represent actions that do not produce values, including
//! assignments, control flow, memory stores, and synchronization.

use super::expr::{Expr, MemoryScope};
use super::types::AddressSpace;

/// HIR statement representing an action in a kernel body.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// Variable assignment.
    Assign {
        /// Target variable name.
        target: String,
        /// Value to assign.
        value: Expr,
    },
    /// Conditional statement with optional else branch.
    If {
        /// Condition expression (must evaluate to bool).
        condition: Expr,
        /// Statements executed when condition is true.
        then_body: Vec<Stmt>,
        /// Optional statements executed when condition is false.
        else_body: Option<Vec<Stmt>>,
    },
    /// For loop with range iteration.
    For {
        /// Loop variable name.
        var: String,
        /// Start of range (inclusive).
        start: Expr,
        /// End of range (exclusive).
        end: Expr,
        /// Step increment.
        step: Expr,
        /// Loop body.
        body: Vec<Stmt>,
    },
    /// While loop.
    While {
        /// Loop condition.
        condition: Expr,
        /// Loop body.
        body: Vec<Stmt>,
    },
    /// Return from kernel.
    Return {
        /// Optional return value (kernels typically return void).
        value: Option<Expr>,
    },
    /// Store value to memory address.
    Store {
        /// Address expression.
        addr: Expr,
        /// Value to store.
        value: Expr,
        /// Address space of the store target.
        space: AddressSpace,
    },
    /// Workgroup barrier synchronization.
    Barrier,
    /// Memory fence with scope.
    Fence {
        /// Scope of the fence.
        scope: MemoryScope,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::expr::{BinOp, Literal};

    #[test]
    fn test_assign_stmt() {
        let stmt = Stmt::Assign {
            target: "x".into(),
            value: Expr::Literal(Literal::Int(42)),
        };
        match &stmt {
            Stmt::Assign { target, value } => {
                assert_eq!(target, "x");
                assert_eq!(*value, Expr::Literal(Literal::Int(42)));
            }
            _ => panic!("expected Assign"),
        }
    }

    #[test]
    fn test_if_stmt_with_else() {
        let stmt = Stmt::If {
            condition: Expr::BinOp {
                op: BinOp::Lt,
                lhs: Box::new(Expr::Var("gid".into())),
                rhs: Box::new(Expr::Var("n".into())),
            },
            then_body: vec![Stmt::Assign {
                target: "x".into(),
                value: Expr::Literal(Literal::Int(1)),
            }],
            else_body: Some(vec![Stmt::Assign {
                target: "x".into(),
                value: Expr::Literal(Literal::Int(0)),
            }]),
        };
        match &stmt {
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                assert_eq!(then_body.len(), 1);
                assert!(else_body.is_some());
                assert_eq!(else_body.as_ref().unwrap().len(), 1);
            }
            _ => panic!("expected If"),
        }
    }

    #[test]
    fn test_for_loop() {
        let stmt = Stmt::For {
            var: "i".into(),
            start: Expr::Literal(Literal::Int(0)),
            end: Expr::Literal(Literal::Int(10)),
            step: Expr::Literal(Literal::Int(1)),
            body: vec![Stmt::Barrier],
        };
        match &stmt {
            Stmt::For { var, body, .. } => {
                assert_eq!(var, "i");
                assert_eq!(body.len(), 1);
            }
            _ => panic!("expected For"),
        }
    }
}
