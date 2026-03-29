// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! HIR kernel definition for WAVE GPU kernels.
//!
//! A kernel is the top-level compilation unit, containing parameters,
//! body statements, and optional attributes that control compilation.

use super::stmt::Stmt;
use super::types::{AddressSpace, Type};

/// A kernel parameter with name, type, and address space.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelParam {
    /// Parameter name.
    pub name: String,
    /// Parameter type.
    pub ty: Type,
    /// Address space for pointer parameters.
    pub address_space: AddressSpace,
}

/// Attributes controlling kernel compilation behavior.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct KernelAttributes {
    /// Requested workgroup size [x, y, z].
    pub workgroup_size: Option<[u32; 3]>,
    /// Maximum number of registers to use.
    pub max_registers: Option<u32>,
}

/// A GPU kernel definition.
#[derive(Debug, Clone, PartialEq)]
pub struct Kernel {
    /// Kernel name.
    pub name: String,
    /// Kernel parameters.
    pub params: Vec<KernelParam>,
    /// Kernel body.
    pub body: Vec<Stmt>,
    /// Kernel attributes.
    pub attributes: KernelAttributes,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::expr::{BinOp, Dimension, Expr, Literal};

    #[test]
    fn test_kernel_construction() {
        let kernel = Kernel {
            name: "vector_add".into(),
            params: vec![
                KernelParam {
                    name: "a".into(),
                    ty: Type::Ptr(AddressSpace::Device),
                    address_space: AddressSpace::Device,
                },
                KernelParam {
                    name: "b".into(),
                    ty: Type::Ptr(AddressSpace::Device),
                    address_space: AddressSpace::Device,
                },
                KernelParam {
                    name: "out".into(),
                    ty: Type::Ptr(AddressSpace::Device),
                    address_space: AddressSpace::Device,
                },
                KernelParam {
                    name: "n".into(),
                    ty: Type::U32,
                    address_space: AddressSpace::Private,
                },
            ],
            body: vec![Stmt::Assign {
                target: "gid".into(),
                value: Expr::ThreadId(Dimension::X),
            }],
            attributes: KernelAttributes::default(),
        };
        assert_eq!(kernel.name, "vector_add");
        assert_eq!(kernel.params.len(), 4);
        assert_eq!(kernel.body.len(), 1);
    }

    #[test]
    fn test_kernel_with_if_body() {
        let kernel = Kernel {
            name: "guarded_add".into(),
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
            attributes: KernelAttributes {
                workgroup_size: Some([256, 1, 1]),
                max_registers: Some(32),
            },
        };
        assert_eq!(kernel.body.len(), 2);
        assert_eq!(kernel.attributes.workgroup_size, Some([256, 1, 1]));
        assert_eq!(kernel.attributes.max_registers, Some(32));
    }
}
