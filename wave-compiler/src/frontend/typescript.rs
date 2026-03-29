// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! TypeScript kernel parser producing HIR.
//!
//! Parses a subset of TypeScript suitable for GPU kernels using
//! line-based parsing. Supports function definitions with type
//! annotations, arithmetic, if/else, and array indexing.

use crate::diagnostics::CompileError;
use crate::hir::expr::{BinOp, Dimension, Expr, Literal};
use crate::hir::kernel::{Kernel, KernelAttributes, KernelParam};
use crate::hir::stmt::Stmt;
use crate::hir::types::{AddressSpace, Type};

/// Parse a TypeScript kernel source string into an HIR Kernel.
///
/// # Errors
///
/// Returns `CompileError::ParseError` if the source cannot be parsed.
pub fn parse_typescript(source: &str) -> Result<Kernel, CompileError> {
    let lines: Vec<&str> = source.lines().collect();
    let mut parser = TsParser::new(&lines);
    parser.parse_kernel()
}

struct TsParser<'a> {
    lines: &'a [&'a str],
    pos: usize,
}

impl<'a> TsParser<'a> {
    fn new(lines: &'a [&'a str]) -> Self {
        Self { lines, pos: 0 }
    }

    fn parse_kernel(&mut self) -> Result<Kernel, CompileError> {
        while self.pos < self.lines.len() {
            let line = self.lines[self.pos].trim();
            if line.starts_with("function ") || line.contains("function ") || line.starts_with("export function") {
                return self.parse_function();
            }
            if line.starts_with("import ") || line.is_empty() || line.starts_with("//") || line.starts_with("kernel(") {
                self.pos += 1;
                continue;
            }
            self.pos += 1;
        }
        Err(CompileError::ParseError {
            message: "no kernel function found".into(),
        })
    }

    fn parse_function(&mut self) -> Result<Kernel, CompileError> {
        let line = self.lines[self.pos].trim().to_string();
        let func_pos = line.find("function ").ok_or_else(|| CompileError::ParseError {
            message: "expected 'function'".into(),
        })?;
        let after_func = &line[func_pos + 9..];
        let paren_start = after_func.find('(').ok_or_else(|| CompileError::ParseError {
            message: "expected '('".into(),
        })?;
        let name = after_func[..paren_start].trim().to_string();

        let paren_end = line.find(')').ok_or_else(|| CompileError::ParseError {
            message: "expected ')'".into(),
        })?;
        let params_str = &line[line.find('(').unwrap() + 1..paren_end];
        let params = self.parse_params(params_str)?;

        self.pos += 1;

        let body = self.parse_body()?;

        Ok(Kernel {
            name,
            params,
            body,
            attributes: KernelAttributes::default(),
        })
    }

    fn parse_params(&self, s: &str) -> Result<Vec<KernelParam>, CompileError> {
        let mut params = Vec::new();
        for param in s.split(',') {
            let param = param.trim();
            if param.is_empty() {
                continue;
            }
            let parts: Vec<&str> = param.splitn(2, ':').collect();
            let name = parts[0].trim().to_string();
            let (ty, space) = if parts.len() > 1 {
                let type_str = parts[1].trim();
                if type_str.contains("[]") {
                    (Type::Ptr(AddressSpace::Device), AddressSpace::Device)
                } else {
                    match type_str {
                        "u32" | "number" => (Type::U32, AddressSpace::Private),
                        "i32" => (Type::I32, AddressSpace::Private),
                        "f32" => (Type::F32, AddressSpace::Private),
                        "f64" => (Type::F64, AddressSpace::Private),
                        "boolean" => (Type::Bool, AddressSpace::Private),
                        _ => (Type::U32, AddressSpace::Private),
                    }
                }
            } else {
                (Type::U32, AddressSpace::Private)
            };
            params.push(KernelParam {
                name,
                ty,
                address_space: space,
            });
        }
        Ok(params)
    }

    fn parse_body(&mut self) -> Result<Vec<Stmt>, CompileError> {
        let mut stmts = Vec::new();
        let mut brace_depth = 0i32;

        while self.pos < self.lines.len() {
            let line = self.lines[self.pos].trim();
            if line == "{" {
                brace_depth += 1;
                self.pos += 1;
                continue;
            }
            if line == "}" || line == "});" || line.starts_with("})") {
                if brace_depth <= 0 {
                    self.pos += 1;
                    break;
                }
                brace_depth -= 1;
                self.pos += 1;
                continue;
            }
            if line.is_empty() || line.starts_with("//") {
                self.pos += 1;
                continue;
            }

            if line.starts_with("if ") || line.starts_with("if(") {
                stmts.push(self.parse_if()?);
            } else if line.starts_with("const ") || line.starts_with("let ") || line.starts_with("var ") {
                stmts.push(self.parse_declaration()?);
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
        let condition = parse_ts_expr(cond_str)?;
        self.pos += 1;

        let then_body = self.parse_body()?;

        let else_body = if self.pos < self.lines.len() && self.lines[self.pos].trim().starts_with("else") {
            self.pos += 1;
            Some(self.parse_body()?)
        } else {
            None
        };

        Ok(Stmt::If { condition, then_body, else_body })
    }

    fn parse_declaration(&mut self) -> Result<Stmt, CompileError> {
        let line = self.lines[self.pos].trim().trim_end_matches(';');
        self.pos += 1;

        let clean = line
            .trim_start_matches("const ")
            .trim_start_matches("let ")
            .trim_start_matches("var ");

        let eq_pos = clean.find('=').ok_or_else(|| CompileError::ParseError {
            message: format!("expected '=' in declaration: {line}"),
        })?;

        let lhs = clean[..eq_pos].trim();
        let target = lhs.split(':').next().unwrap_or(lhs).trim().to_string();
        let value = parse_ts_expr(clean[eq_pos + 1..].trim())?;

        Ok(Stmt::Assign { target, value })
    }

    fn parse_assignment(&mut self) -> Result<Stmt, CompileError> {
        let line = self.lines[self.pos].trim().trim_end_matches(';');
        self.pos += 1;

        let eq_pos = line.find('=').ok_or_else(|| CompileError::ParseError {
            message: format!("expected '=' in assignment: {line}"),
        })?;

        let lhs = line[..eq_pos].trim();
        let rhs = line[eq_pos + 1..].trim();
        let value = parse_ts_expr(rhs)?;

        if lhs.contains('[') {
            let bracket_pos = lhs.find('[').unwrap();
            let bracket_end = lhs.find(']').unwrap();
            let base_name = lhs[..bracket_pos].trim();
            let index_str = &lhs[bracket_pos + 1..bracket_end];
            let base = Expr::Var(base_name.to_string());
            let index = parse_ts_expr(index_str)?;
            let offset = Expr::BinOp { op: BinOp::Mul, lhs: Box::new(index), rhs: Box::new(Expr::Literal(Literal::Int(4))) };
            let addr = Expr::BinOp { op: BinOp::Add, lhs: Box::new(base), rhs: Box::new(offset) };
            return Ok(Stmt::Store { addr, value, space: AddressSpace::Device });
        }

        Ok(Stmt::Assign { target: lhs.to_string(), value })
    }
}

fn parse_ts_expr(s: &str) -> Result<Expr, CompileError> {
    let s = s.trim();

    for &(op_str, op) in &[(" + ", BinOp::Add), (" - ", BinOp::Sub)] {
        if let Some(pos) = s.rfind(op_str) {
            return Ok(Expr::BinOp {
                op,
                lhs: Box::new(parse_ts_expr(&s[..pos])?),
                rhs: Box::new(parse_ts_expr(&s[pos + op_str.len()..])?),
            });
        }
    }

    for &(op_str, op) in &[(" * ", BinOp::Mul), (" / ", BinOp::Div)] {
        if let Some(pos) = s.rfind(op_str) {
            return Ok(Expr::BinOp {
                op,
                lhs: Box::new(parse_ts_expr(&s[..pos])?),
                rhs: Box::new(parse_ts_expr(&s[pos + op_str.len()..])?),
            });
        }
    }

    for &(op_str, op) in &[(" < ", BinOp::Lt), (" > ", BinOp::Gt), (" === ", BinOp::Eq), (" !== ", BinOp::Ne)] {
        if let Some(pos) = s.rfind(op_str) {
            return Ok(Expr::BinOp {
                op,
                lhs: Box::new(parse_ts_expr(&s[..pos])?),
                rhs: Box::new(parse_ts_expr(&s[pos + op_str.len()..])?),
            });
        }
    }

    if s.starts_with('(') && s.ends_with(')') {
        return parse_ts_expr(&s[1..s.len() - 1]);
    }

    match s {
        "threadId()" | "thread_id()" => return Ok(Expr::ThreadId(Dimension::X)),
        "workgroupId()" => return Ok(Expr::WorkgroupId(Dimension::X)),
        _ => {}
    }

    if let Some("threadId" | "thread_id") = s.strip_suffix("()") {
        return Ok(Expr::ThreadId(Dimension::X));
    }

    if let Some(bracket_pos) = s.find('[') {
        if s.ends_with(']') {
            return Ok(Expr::Index {
                base: Box::new(parse_ts_expr(&s[..bracket_pos])?),
                index: Box::new(parse_ts_expr(&s[bracket_pos + 1..s.len() - 1])?),
            });
        }
    }

    if let Ok(v) = s.parse::<i64>() { return Ok(Expr::Literal(Literal::Int(v))); }
    if let Ok(v) = s.parse::<f64>() { return Ok(Expr::Literal(Literal::Float(v))); }

    if s.chars().all(|c| c.is_alphanumeric() || c == '_') && !s.is_empty() {
        return Ok(Expr::Var(s.to_string()));
    }

    Err(CompileError::ParseError {
        message: format!("cannot parse TS expression: '{s}'"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ts_vector_add() {
        let source = r#"
import { kernel, f32, threadId } from "wave";

function vectorAdd(a: f32[], b: f32[], out: f32[], n: u32) {
    const gid = threadId();
    if (gid < n) {
        out[gid] = a[gid] + b[gid];
    }
}
"#;
        let kernel = parse_typescript(source).unwrap();
        assert_eq!(kernel.name, "vectorAdd");
        assert_eq!(kernel.params.len(), 4);
        assert!(kernel.params[0].ty.is_pointer());
    }

    #[test]
    fn test_parse_ts_simple() {
        let source = r#"
function test(n: u32) {
    const x = 42;
}
"#;
        let kernel = parse_typescript(source).unwrap();
        assert_eq!(kernel.name, "test");
        assert_eq!(kernel.body.len(), 1);
    }
}
