// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! C/C++ kernel parser producing HIR.
//!
//! Parses a subset of C/C++ suitable for GPU kernels using line-based
//! parsing for the restricted kernel subset. Supports `__kernel` functions,
//! basic types, arithmetic, if/else, and array indexing.

use crate::diagnostics::CompileError;
use crate::hir::expr::{BinOp, Dimension, Expr, Literal};
use crate::hir::kernel::{Kernel, KernelAttributes, KernelParam};
use crate::hir::stmt::Stmt;
use crate::hir::types::{AddressSpace, Type};

/// Parse a C/C++ kernel source string into an HIR Kernel.
///
/// # Errors
///
/// Returns `CompileError::ParseError` if the source cannot be parsed.
pub fn parse_cpp(source: &str) -> Result<Kernel, CompileError> {
    let lines: Vec<&str> = source.lines().collect();
    let mut parser = CppParser::new(&lines);
    parser.parse_kernel()
}

struct CppParser<'a> {
    lines: &'a [&'a str],
    pos: usize,
}

impl<'a> CppParser<'a> {
    fn new(lines: &'a [&'a str]) -> Self {
        Self { lines, pos: 0 }
    }

    fn parse_kernel(&mut self) -> Result<Kernel, CompileError> {
        while self.pos < self.lines.len() {
            let line = self.lines[self.pos].trim();
            if line.contains("__kernel") || line.contains("void ") || line.contains("__global__") {
                return self.parse_function();
            }
            self.pos += 1;
        }
        Err(CompileError::ParseError {
            message: "no kernel function found".into(),
        })
    }

    fn parse_function(&mut self) -> Result<Kernel, CompileError> {
        let line = self.lines[self.pos].trim().to_string();
        let paren_start = line.find('(').ok_or_else(|| CompileError::ParseError {
            message: "expected '(' in function definition".into(),
        })?;
        let paren_end = line.find(')').ok_or_else(|| CompileError::ParseError {
            message: "expected ')' in function definition".into(),
        })?;

        let before_paren = &line[..paren_start];
        let name = before_paren
            .split_whitespace()
            .last()
            .unwrap_or("kernel")
            .to_string();

        let params_str = &line[paren_start + 1..paren_end];
        let params = Self::parse_params(params_str);

        self.pos += 1;
        while self.pos < self.lines.len() && self.lines[self.pos].trim() == "{" {
            self.pos += 1;
        }

        let body = self.parse_body()?;

        Ok(Kernel {
            name,
            params,
            body,
            attributes: KernelAttributes::default(),
        })
    }

    fn parse_params(s: &str) -> Vec<KernelParam> {
        let mut params = Vec::new();
        for param in s.split(',') {
            let param = param.trim();
            if param.is_empty() {
                continue;
            }
            let parts: Vec<&str> = param.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }
            let name_part = parts.last().unwrap().trim_start_matches('*');
            let is_pointer = param.contains('*');
            let type_str = parts[0];

            let (ty, space) = if is_pointer {
                (Type::Ptr(AddressSpace::Device), AddressSpace::Device)
            } else {
                match type_str {
                    "float" | "f32" => (Type::F32, AddressSpace::Private),
                    "int" | "i32" => (Type::I32, AddressSpace::Private),
                    "double" | "f64" => (Type::F64, AddressSpace::Private),
                    "bool" => (Type::Bool, AddressSpace::Private),
                    _ => (Type::U32, AddressSpace::Private),
                }
            };

            params.push(KernelParam {
                name: name_part.to_string(),
                ty,
                address_space: space,
            });
        }
        params
    }

    fn parse_body(&mut self) -> Result<Vec<Stmt>, CompileError> {
        let mut stmts = Vec::new();
        while self.pos < self.lines.len() {
            let line = self.lines[self.pos].trim();
            if line == "}" {
                self.pos += 1;
                break;
            }
            if line.is_empty() || line.starts_with("//") {
                self.pos += 1;
                continue;
            }
            if line.starts_with("if ") || line.starts_with("if(") {
                stmts.push(self.parse_if()?);
            } else if line.contains('=') && !line.contains("==") && !line.contains("!=") {
                stmts.push(self.parse_assignment()?);
            } else {
                self.pos += 1;
            }
        }
        Ok(stmts)
    }

    fn parse_if(&mut self) -> Result<Stmt, CompileError> {
        let line = self.lines[self.pos].trim();
        let cond_start = line.find('(').unwrap_or(3);
        let cond_end = line.rfind(')').unwrap_or(line.len());
        let cond_str = &line[cond_start + 1..cond_end];
        let condition = parse_c_expr(cond_str)?;
        self.pos += 1;

        while self.pos < self.lines.len() && self.lines[self.pos].trim() == "{" {
            self.pos += 1;
        }

        let then_body = self.parse_body()?;

        let else_body = if self.pos < self.lines.len() {
            let next = self.lines[self.pos].trim();
            if next.starts_with("else") || next == "} else {" {
                self.pos += 1;
                while self.pos < self.lines.len() && self.lines[self.pos].trim() == "{" {
                    self.pos += 1;
                }
                Some(self.parse_body()?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Stmt::If {
            condition,
            then_body,
            else_body,
        })
    }

    fn parse_assignment(&mut self) -> Result<Stmt, CompileError> {
        let line = self.lines[self.pos].trim().trim_end_matches(';');
        self.pos += 1;

        let parts: Vec<&str> = line.splitn(2, '=').collect();
        if parts.len() != 2 {
            return Err(CompileError::ParseError {
                message: format!("invalid assignment: {line}"),
            });
        }

        let lhs = parts[0].trim();
        let rhs = parts[1].trim();

        let lhs_clean = lhs
            .trim_start_matches("uint32_t ")
            .trim_start_matches("int ")
            .trim_start_matches("float ")
            .trim_start_matches("double ")
            .trim_start_matches("auto ")
            .trim();

        let value = parse_c_expr(rhs)?;

        if lhs_clean.contains('[') {
            let bracket_pos = lhs_clean.find('[').unwrap();
            let bracket_end = lhs_clean.find(']').unwrap();
            let base_name = lhs_clean[..bracket_pos].trim();
            let index_str = &lhs_clean[bracket_pos + 1..bracket_end];

            let base = Expr::Var(base_name.to_string());
            let index = parse_c_expr(index_str)?;
            let offset = Expr::BinOp {
                op: BinOp::Mul,
                lhs: Box::new(index),
                rhs: Box::new(Expr::Literal(Literal::Int(4))),
            };
            let addr = Expr::BinOp {
                op: BinOp::Add,
                lhs: Box::new(base),
                rhs: Box::new(offset),
            };
            return Ok(Stmt::Store {
                addr,
                value,
                space: AddressSpace::Device,
            });
        }

        Ok(Stmt::Assign {
            target: lhs_clean.to_string(),
            value,
        })
    }
}

fn parse_c_expr(s: &str) -> Result<Expr, CompileError> {
    let s = s.trim();

    for &(op_str, op) in &[(" + ", BinOp::Add), (" - ", BinOp::Sub)] {
        if let Some(pos) = s.rfind(op_str) {
            let lhs = parse_c_expr(&s[..pos])?;
            let rhs = parse_c_expr(&s[pos + op_str.len()..])?;
            return Ok(Expr::BinOp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            });
        }
    }

    for &(op_str, op) in &[
        (" * ", BinOp::Mul),
        (" / ", BinOp::Div),
        (" % ", BinOp::Mod),
    ] {
        if let Some(pos) = s.rfind(op_str) {
            let lhs = parse_c_expr(&s[..pos])?;
            let rhs = parse_c_expr(&s[pos + op_str.len()..])?;
            return Ok(Expr::BinOp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            });
        }
    }

    for &(op_str, op) in &[
        (" < ", BinOp::Lt),
        (" <= ", BinOp::Le),
        (" > ", BinOp::Gt),
        (" >= ", BinOp::Ge),
        (" == ", BinOp::Eq),
        (" != ", BinOp::Ne),
    ] {
        if let Some(pos) = s.rfind(op_str) {
            let lhs = parse_c_expr(&s[..pos])?;
            let rhs = parse_c_expr(&s[pos + op_str.len()..])?;
            return Ok(Expr::BinOp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            });
        }
    }

    if s.starts_with('(') && s.ends_with(')') {
        return parse_c_expr(&s[1..s.len() - 1]);
    }

    match s {
        "thread_id()" => return Ok(Expr::ThreadId(Dimension::X)),
        "workgroup_id()" => return Ok(Expr::WorkgroupId(Dimension::X)),
        _ => {}
    }

    if let Some(bracket_pos) = s.find('[') {
        if s.ends_with(']') {
            let base = &s[..bracket_pos];
            let index = &s[bracket_pos + 1..s.len() - 1];
            return Ok(Expr::Index {
                base: Box::new(parse_c_expr(base)?),
                index: Box::new(parse_c_expr(index)?),
            });
        }
    }

    if let Ok(v) = s.parse::<i64>() {
        return Ok(Expr::Literal(Literal::Int(v)));
    }
    if let Ok(v) = s.parse::<f64>() {
        return Ok(Expr::Literal(Literal::Float(v)));
    }

    if s.chars().all(|c| c.is_alphanumeric() || c == '_') && !s.is_empty() {
        return Ok(Expr::Var(s.to_string()));
    }

    Err(CompileError::ParseError {
        message: format!("cannot parse C expression: '{s}'"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cpp_vector_add() {
        let source = r#"
__kernel void vector_add(float* a, float* b, float* out, uint32_t n) {
    uint32_t gid = thread_id();
    if (gid < n) {
        out[gid] = a[gid] + b[gid];
    }
}
"#;
        let kernel = parse_cpp(source).unwrap();
        assert_eq!(kernel.name, "vector_add");
        assert_eq!(kernel.params.len(), 4);
        assert!(kernel.params[0].ty.is_pointer());
        assert_eq!(kernel.params[3].ty, Type::U32);
    }

    #[test]
    fn test_parse_cpp_simple() {
        let source = r#"
void test(uint32_t n) {
    uint32_t x = 42;
}
"#;
        let kernel = parse_cpp(source).unwrap();
        assert_eq!(kernel.name, "test");
        assert_eq!(kernel.body.len(), 1);
    }
}
