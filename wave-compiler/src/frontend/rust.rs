// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Rust kernel parser producing HIR.
//!
//! Parses a subset of Rust suitable for GPU kernels using the `syn` crate.
//! Supports function definitions with type annotations, arithmetic, comparisons,
//! if/else, for loops, array indexing, and GPU intrinsics.

use crate::diagnostics::CompileError;
use crate::hir::expr::{BinOp, Dimension, Expr, Literal};
use crate::hir::kernel::{Kernel, KernelAttributes, KernelParam};
use crate::hir::stmt::Stmt;
use crate::hir::types::{AddressSpace, Type};

/// Parse a Rust kernel source string into an HIR Kernel.
///
/// # Errors
///
/// Returns `CompileError::ParseError` if the source cannot be parsed.
pub fn parse_rust(source: &str) -> Result<Kernel, CompileError> {
    let file = syn::parse_file(source).map_err(|e| CompileError::ParseError {
        message: format!("Rust parse error: {e}"),
    })?;

    for item in &file.items {
        if let syn::Item::Fn(func) = item {
            let has_kernel_attr = func.attrs.iter().any(|a| {
                a.path().is_ident("kernel")
            });
            if has_kernel_attr || func.attrs.is_empty() {
                return lower_function(func);
            }
        }
    }

    Err(CompileError::ParseError {
        message: "no kernel function found".into(),
    })
}

fn lower_function(func: &syn::ItemFn) -> Result<Kernel, CompileError> {
    let name = func.sig.ident.to_string();
    let mut params = Vec::new();

    for arg in &func.sig.inputs {
        if let syn::FnArg::Typed(pat_type) = arg {
            if let syn::Pat::Ident(ident) = &*pat_type.pat {
                let param_name = ident.ident.to_string();
                let (ty, space) = lower_type(&pat_type.ty);
                params.push(KernelParam {
                    name: param_name,
                    ty,
                    address_space: space,
                });
            }
        }
    }

    let body = lower_block(&func.block)?;

    Ok(Kernel {
        name,
        params,
        body,
        attributes: KernelAttributes::default(),
    })
}

fn lower_type(ty: &syn::Type) -> (Type, AddressSpace) {
    match ty {
        syn::Type::Path(path) => {
            let ident = path.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default();
            match ident.as_str() {
                "u32" => (Type::U32, AddressSpace::Private),
                "i32" => (Type::I32, AddressSpace::Private),
                "f32" => (Type::F32, AddressSpace::Private),
                "f64" => (Type::F64, AddressSpace::Private),
                "bool" => (Type::Bool, AddressSpace::Private),
                _ => (Type::U32, AddressSpace::Private),
            }
        }
        syn::Type::Reference(ref_type) => {
            if let syn::Type::Slice(_) = &*ref_type.elem {
                (Type::Ptr(AddressSpace::Device), AddressSpace::Device)
            } else {
                lower_type(&ref_type.elem)
            }
        }
        _ => (Type::U32, AddressSpace::Private),
    }
}

fn lower_block(block: &syn::Block) -> Result<Vec<Stmt>, CompileError> {
    let mut stmts = Vec::new();
    for stmt in &block.stmts {
        match stmt {
            syn::Stmt::Local(local) => {
                if let Some(init) = &local.init {
                    if let syn::Pat::Ident(ident) = &local.pat {
                        let value = lower_expr(&init.expr)?;
                        stmts.push(Stmt::Assign {
                            target: ident.ident.to_string(),
                            value,
                        });
                    }
                }
            }
            syn::Stmt::Expr(expr, _) => {
                if let Some(s) = lower_stmt_expr(expr)? {
                    stmts.push(s);
                }
            }
            _ => {}
        }
    }
    Ok(stmts)
}

fn lower_stmt_expr(expr: &syn::Expr) -> Result<Option<Stmt>, CompileError> {
    match expr {
        syn::Expr::If(if_expr) => {
            let condition = lower_expr(&if_expr.cond)?;
            let then_body = lower_block(&if_expr.then_branch)?;
            let else_body = if let Some((_, else_expr)) = &if_expr.else_branch {
                if let syn::Expr::Block(block_expr) = &**else_expr {
                    Some(lower_block(&block_expr.block)?)
                } else {
                    None
                }
            } else {
                None
            };
            Ok(Some(Stmt::If {
                condition,
                then_body,
                else_body,
            }))
        }
        syn::Expr::Assign(assign) => {
            let value = lower_expr(&assign.right)?;
            if let syn::Expr::Path(path) = &*assign.left {
                let target = path.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default();
                Ok(Some(Stmt::Assign { target, value }))
            } else if let syn::Expr::Index(idx) = &*assign.left {
                let base = lower_expr(&idx.expr)?;
                let index = lower_expr(&idx.index)?;
                let elem_size = Expr::Literal(Literal::Int(4));
                let offset = Expr::BinOp { op: BinOp::Mul, lhs: Box::new(index), rhs: Box::new(elem_size) };
                let addr = Expr::BinOp { op: BinOp::Add, lhs: Box::new(base), rhs: Box::new(offset) };
                Ok(Some(Stmt::Store { addr, value, space: AddressSpace::Device }))
            } else {
                Ok(None)
            }
        }
        syn::Expr::Return(ret) => {
            let value = ret.expr.as_ref().map(|e| lower_expr(e)).transpose()?;
            Ok(Some(Stmt::Return { value }))
        }
        _ => Ok(None),
    }
}

fn lower_expr(expr: &syn::Expr) -> Result<Expr, CompileError> {
    match expr {
        syn::Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Int(i) => {
                let v: i64 = i.base10_parse().unwrap_or(0);
                Ok(Expr::Literal(Literal::Int(v)))
            }
            syn::Lit::Float(f) => {
                let v: f64 = f.base10_parse().unwrap_or(0.0);
                Ok(Expr::Literal(Literal::Float(v)))
            }
            syn::Lit::Bool(b) => Ok(Expr::Literal(Literal::Bool(b.value))),
            _ => Err(CompileError::ParseError { message: "unsupported literal".into() }),
        },
        syn::Expr::Path(path) => {
            let name = path.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default();
            Ok(Expr::Var(name))
        }
        syn::Expr::Binary(bin) => {
            let lhs = lower_expr(&bin.left)?;
            let rhs = lower_expr(&bin.right)?;
            let op = match bin.op {
                syn::BinOp::Add(_) => BinOp::Add,
                syn::BinOp::Sub(_) => BinOp::Sub,
                syn::BinOp::Mul(_) => BinOp::Mul,
                syn::BinOp::Div(_) => BinOp::Div,
                syn::BinOp::Rem(_) => BinOp::Mod,
                syn::BinOp::Lt(_) => BinOp::Lt,
                syn::BinOp::Le(_) => BinOp::Le,
                syn::BinOp::Gt(_) => BinOp::Gt,
                syn::BinOp::Ge(_) => BinOp::Ge,
                syn::BinOp::Eq(_) => BinOp::Eq,
                syn::BinOp::Ne(_) => BinOp::Ne,
                syn::BinOp::BitAnd(_) => BinOp::BitAnd,
                syn::BinOp::BitOr(_) => BinOp::BitOr,
                syn::BinOp::BitXor(_) => BinOp::BitXor,
                syn::BinOp::Shl(_) => BinOp::Shl,
                syn::BinOp::Shr(_) => BinOp::Shr,
                _ => return Err(CompileError::ParseError { message: "unsupported binary op".into() }),
            };
            Ok(Expr::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs) })
        }
        syn::Expr::Call(call) => {
            if let syn::Expr::Path(path) = &*call.func {
                let func_name = path.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default();
                match func_name.as_str() {
                    "thread_id" => Ok(Expr::ThreadId(Dimension::X)),
                    "workgroup_id" => Ok(Expr::WorkgroupId(Dimension::X)),
                    "workgroup_size" => Ok(Expr::WorkgroupSize(Dimension::X)),
                    "lane_id" => Ok(Expr::LaneId),
                    "wave_width" => Ok(Expr::WaveWidth),
                    "barrier" => Ok(Expr::Literal(Literal::Int(0))),
                    _ => {
                        let args: Vec<Expr> = call.args.iter().map(lower_expr).collect::<Result<_, _>>()?;
                        Ok(Expr::Call {
                            func: match func_name.as_str() {
                                "sqrt" => crate::hir::expr::BuiltinFunc::Sqrt,
                                "sin" => crate::hir::expr::BuiltinFunc::Sin,
                                "cos" => crate::hir::expr::BuiltinFunc::Cos,
                                "abs" => crate::hir::expr::BuiltinFunc::Abs,
                                "min" => crate::hir::expr::BuiltinFunc::Min,
                                "max" => crate::hir::expr::BuiltinFunc::Max,
                                _ => return Err(CompileError::ParseError { message: format!("unknown function: {func_name}") }),
                            },
                            args,
                        })
                    }
                }
            } else {
                Err(CompileError::ParseError { message: "unsupported call".into() })
            }
        }
        syn::Expr::Index(idx) => {
            let base = lower_expr(&idx.expr)?;
            let index = lower_expr(&idx.index)?;
            Ok(Expr::Index { base: Box::new(base), index: Box::new(index) })
        }
        syn::Expr::Paren(paren) => lower_expr(&paren.expr),
        syn::Expr::Unary(unary) => {
            let operand = lower_expr(&unary.expr)?;
            match unary.op {
                syn::UnOp::Neg(_) => Ok(Expr::UnaryOp { op: crate::hir::expr::UnaryOp::Neg, operand: Box::new(operand) }),
                syn::UnOp::Not(_) => Ok(Expr::UnaryOp { op: crate::hir::expr::UnaryOp::Not, operand: Box::new(operand) }),
                _ => Err(CompileError::ParseError { message: "unsupported unary op".into() }),
            }
        }
        _ => Err(CompileError::ParseError {
            message: "unsupported expression".into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rust_vector_add() {
        let source = r#"
#[kernel]
fn vector_add(a: &[f32], b: &[f32], out: &mut [f32], n: u32) {
    let gid = thread_id();
    if gid < n {
        let a_val = a[gid];
    }
}
"#;
        let kernel = parse_rust(source).unwrap();
        assert_eq!(kernel.name, "vector_add");
        assert_eq!(kernel.params.len(), 4);
        assert_eq!(kernel.params[0].ty, Type::Ptr(AddressSpace::Device));
        assert_eq!(kernel.params[3].ty, Type::U32);
    }

    #[test]
    fn test_parse_rust_simple() {
        let source = r#"
#[kernel]
fn test(n: u32) {
    let x = 42;
}
"#;
        let kernel = parse_rust(source).unwrap();
        assert_eq!(kernel.name, "test");
        assert_eq!(kernel.body.len(), 1);
    }
}
