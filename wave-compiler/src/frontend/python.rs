// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Python kernel parser producing HIR.
//!
//! Parses a subset of Python suitable for GPU kernels: function definitions
//! with type annotations, arithmetic, comparisons, if/else, for range(),
//! array indexing, and GPU intrinsics (thread_id, barrier, etc.).
//! Uses line-by-line parsing for the restricted kernel subset.

use crate::diagnostics::CompileError;
use crate::hir::expr::{BinOp, BuiltinFunc, Dimension, Expr, Literal, UnaryOp};
use crate::hir::kernel::{Kernel, KernelAttributes, KernelParam};
use crate::hir::stmt::Stmt;
use crate::hir::types::{AddressSpace, Type};

/// Parse a Python kernel source string into an HIR Kernel.
///
/// # Errors
///
/// Returns `CompileError::ParseError` if the source cannot be parsed.
pub fn parse_python(source: &str) -> Result<Kernel, CompileError> {
    let lines: Vec<&str> = source.lines().collect();
    let mut parser = PythonParser::new(&lines);
    parser.parse_kernel()
}

struct PythonParser<'a> {
    lines: &'a [&'a str],
    pos: usize,
}

impl<'a> PythonParser<'a> {
    fn new(lines: &'a [&'a str]) -> Self {
        Self { lines, pos: 0 }
    }

    fn parse_kernel(&mut self) -> Result<Kernel, CompileError> {
        while self.pos < self.lines.len() {
            let line = self.lines[self.pos].trim();
            if line.is_empty()
                || line.starts_with('#')
                || line.starts_with("from ")
                || line.starts_with("import ")
            {
                self.pos += 1;
                continue;
            }
            if line == "@kernel" {
                self.pos += 1;
                continue;
            }
            if line.starts_with("def ") {
                return self.parse_def();
            }
            self.pos += 1;
        }
        Err(CompileError::ParseError {
            message: "no kernel function found".into(),
        })
    }

    fn parse_def(&mut self) -> Result<Kernel, CompileError> {
        let line = self.lines[self.pos].trim();
        let after_def = line
            .strip_prefix("def ")
            .ok_or_else(|| CompileError::ParseError {
                message: "expected 'def'".into(),
            })?;

        let paren_start = after_def
            .find('(')
            .ok_or_else(|| CompileError::ParseError {
                message: "expected '(' in function definition".into(),
            })?;
        let name = after_def[..paren_start].trim().to_string();

        let paren_end = after_def
            .find(')')
            .ok_or_else(|| CompileError::ParseError {
                message: "expected ')' in function definition".into(),
            })?;
        let params_str = &after_def[paren_start + 1..paren_end];
        let params = self.parse_params(params_str)?;

        self.pos += 1;

        let indent = self.get_body_indent()?;
        let body = self.parse_body(indent)?;

        Ok(Kernel {
            name,
            params,
            body,
            attributes: KernelAttributes::default(),
        })
    }

    fn parse_params(&self, params_str: &str) -> Result<Vec<KernelParam>, CompileError> {
        let mut params = Vec::new();
        for param_str in params_str.split(',') {
            let param_str = param_str.trim();
            if param_str.is_empty() {
                continue;
            }
            let parts: Vec<&str> = param_str.splitn(2, ':').collect();
            let param_name = parts[0].trim().to_string();
            let (ty, addr_space) = if parts.len() > 1 {
                self.parse_type_annotation(parts[1].trim())
            } else {
                (Type::U32, AddressSpace::Private)
            };
            params.push(KernelParam {
                name: param_name,
                ty,
                address_space: addr_space,
            });
        }
        Ok(params)
    }

    fn parse_type_annotation(&self, ann: &str) -> (Type, AddressSpace) {
        match ann {
            "u32" | "int" => (Type::U32, AddressSpace::Private),
            "i32" => (Type::I32, AddressSpace::Private),
            "f32" | "float" => (Type::F32, AddressSpace::Private),
            "f16" => (Type::F16, AddressSpace::Private),
            "f64" => (Type::F64, AddressSpace::Private),
            "bool" => (Type::Bool, AddressSpace::Private),
            s if s.contains("[:]") || s.contains("[]") => {
                (Type::Ptr(AddressSpace::Device), AddressSpace::Device)
            }
            _ => (Type::U32, AddressSpace::Private),
        }
    }

    fn get_body_indent(&self) -> Result<usize, CompileError> {
        if self.pos >= self.lines.len() {
            return Err(CompileError::ParseError {
                message: "expected function body".into(),
            });
        }
        let line = self.lines[self.pos];
        Ok(line.len() - line.trim_start().len())
    }

    fn parse_body(&mut self, indent: usize) -> Result<Vec<Stmt>, CompileError> {
        let mut stmts = Vec::new();
        while self.pos < self.lines.len() {
            let line = self.lines[self.pos];
            if line.trim().is_empty() || line.trim().starts_with('#') {
                self.pos += 1;
                continue;
            }
            let current_indent = line.len() - line.trim_start().len();
            if current_indent < indent {
                break;
            }
            let trimmed = line.trim();
            if trimmed.starts_with("if ") {
                stmts.push(self.parse_if()?);
            } else if trimmed.starts_with("for ") {
                stmts.push(self.parse_for()?);
            } else if trimmed.starts_with("while ") {
                stmts.push(self.parse_while()?);
            } else if trimmed == "return" || trimmed.starts_with("return ") {
                stmts.push(self.parse_return()?);
            } else if trimmed == "barrier()" {
                stmts.push(Stmt::Barrier);
                self.pos += 1;
            } else if trimmed.contains('=') && !trimmed.contains("==") {
                stmts.push(self.parse_assignment()?);
            } else {
                self.pos += 1;
            }
        }
        Ok(stmts)
    }

    fn parse_if(&mut self) -> Result<Stmt, CompileError> {
        let line = self.lines[self.pos].trim();
        let cond_str = line
            .strip_prefix("if ")
            .and_then(|s| s.strip_suffix(':'))
            .ok_or_else(|| CompileError::ParseError {
                message: format!("invalid if statement: {line}"),
            })?;
        let condition = self.parse_expr(cond_str.trim())?;
        self.pos += 1;

        let then_indent = self.get_body_indent()?;
        let then_body = self.parse_body(then_indent)?;

        let else_body = if self.pos < self.lines.len() {
            let next = self.lines[self.pos].trim();
            if next.starts_with("else:") || next.starts_with("elif ") {
                self.pos += 1;
                let else_indent = self.get_body_indent()?;
                Some(self.parse_body(else_indent)?)
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

    fn parse_for(&mut self) -> Result<Stmt, CompileError> {
        let line = self.lines[self.pos].trim();
        let inner = line
            .strip_prefix("for ")
            .and_then(|s| s.strip_suffix(':'))
            .ok_or_else(|| CompileError::ParseError {
                message: format!("invalid for statement: {line}"),
            })?;

        let parts: Vec<&str> = inner.splitn(2, " in ").collect();
        if parts.len() != 2 {
            return Err(CompileError::ParseError {
                message: format!("invalid for statement: {line}"),
            });
        }
        let var = parts[0].trim().to_string();
        let range_str = parts[1].trim();

        let (start, end, step) = self.parse_range(range_str)?;

        self.pos += 1;
        let body_indent = self.get_body_indent()?;
        let body = self.parse_body(body_indent)?;

        Ok(Stmt::For {
            var,
            start,
            end,
            step,
            body,
        })
    }

    fn parse_range(&self, s: &str) -> Result<(Expr, Expr, Expr), CompileError> {
        let inner = s
            .strip_prefix("range(")
            .and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| CompileError::ParseError {
                message: format!("expected range(...), got {s}"),
            })?;

        let args: Vec<&str> = inner.split(',').collect();
        match args.len() {
            1 => Ok((
                Expr::Literal(Literal::Int(0)),
                self.parse_expr(args[0].trim())?,
                Expr::Literal(Literal::Int(1)),
            )),
            2 => Ok((
                self.parse_expr(args[0].trim())?,
                self.parse_expr(args[1].trim())?,
                Expr::Literal(Literal::Int(1)),
            )),
            3 => Ok((
                self.parse_expr(args[0].trim())?,
                self.parse_expr(args[1].trim())?,
                self.parse_expr(args[2].trim())?,
            )),
            _ => Err(CompileError::ParseError {
                message: "range() takes 1-3 arguments".into(),
            }),
        }
    }

    fn parse_while(&mut self) -> Result<Stmt, CompileError> {
        let line = self.lines[self.pos].trim();
        let cond_str = line
            .strip_prefix("while ")
            .and_then(|s| s.strip_suffix(':'))
            .ok_or_else(|| CompileError::ParseError {
                message: format!("invalid while statement: {line}"),
            })?;
        let condition = self.parse_expr(cond_str.trim())?;
        self.pos += 1;

        let body_indent = self.get_body_indent()?;
        let body = self.parse_body(body_indent)?;

        Ok(Stmt::While { condition, body })
    }

    fn parse_return(&mut self) -> Result<Stmt, CompileError> {
        let line = self.lines[self.pos].trim();
        self.pos += 1;
        if line == "return" {
            return Ok(Stmt::Return { value: None });
        }
        let val_str = line.strip_prefix("return ").unwrap_or("");
        if val_str.is_empty() {
            Ok(Stmt::Return { value: None })
        } else {
            Ok(Stmt::Return {
                value: Some(self.parse_expr(val_str)?),
            })
        }
    }

    fn parse_assignment(&mut self) -> Result<Stmt, CompileError> {
        let line = self.lines[self.pos].trim().to_string();
        self.pos += 1;

        if let Some(bracket_pos) = line.find('[') {
            if let Some(eq_pos) = line.find('=') {
                if bracket_pos < eq_pos
                    && !line[..eq_pos].ends_with('!')
                    && !line[..eq_pos].ends_with('<')
                    && !line[..eq_pos].ends_with('>')
                {
                    let base_name = line[..bracket_pos].trim();
                    let bracket_end =
                        line[..eq_pos]
                            .rfind(']')
                            .ok_or_else(|| CompileError::ParseError {
                                message: format!("missing ']' in: {line}"),
                            })?;
                    let index_str = &line[bracket_pos + 1..bracket_end];
                    let value_str = line[eq_pos + 1..].trim();

                    let base = self.parse_expr(base_name)?;
                    let index = self.parse_expr(index_str)?;
                    let value = self.parse_expr(value_str)?;

                    let elem_size = Expr::Literal(Literal::Int(4));
                    let offset = Expr::BinOp {
                        op: BinOp::Mul,
                        lhs: Box::new(index),
                        rhs: Box::new(elem_size),
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
            }
        }

        let eq_pos = line.find('=').ok_or_else(|| CompileError::ParseError {
            message: format!("expected '=' in assignment: {line}"),
        })?;

        if eq_pos > 0
            && (line.as_bytes()[eq_pos - 1] == b'!'
                || line.as_bytes()[eq_pos - 1] == b'<'
                || line.as_bytes()[eq_pos - 1] == b'>')
        {
            return Err(CompileError::ParseError {
                message: format!("unexpected operator in: {line}"),
            });
        }
        if eq_pos + 1 < line.len() && line.as_bytes()[eq_pos + 1] == b'=' {
            return Err(CompileError::ParseError {
                message: format!("comparison in assignment position: {line}"),
            });
        }

        let raw_target = line[..eq_pos].trim();
        let target = if let Some(colon_pos) = raw_target.find(':') {
            raw_target[..colon_pos].trim().to_string()
        } else {
            raw_target.to_string()
        };
        let value_str = line[eq_pos + 1..].trim();
        let value = self.parse_expr(value_str)?;

        Ok(Stmt::Assign { target, value })
    }

    fn parse_expr(&self, s: &str) -> Result<Expr, CompileError> {
        let s = s.trim();

        for &(op_str, op) in &[(" + ", BinOp::Add), (" - ", BinOp::Sub)] {
            if let Some(pos) = find_top_level_op(s, op_str) {
                let lhs = self.parse_expr(&s[..pos])?;
                let rhs = self.parse_expr(&s[pos + op_str.len()..])?;
                return Ok(Expr::BinOp {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                });
            }
        }

        for &(op_str, op) in &[
            (" * ", BinOp::Mul),
            (" // ", BinOp::FloorDiv),
            (" / ", BinOp::Div),
            (" % ", BinOp::Mod),
        ] {
            if let Some(pos) = find_top_level_op(s, op_str) {
                let lhs = self.parse_expr(&s[..pos])?;
                let rhs = self.parse_expr(&s[pos + op_str.len()..])?;
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
            if let Some(pos) = find_top_level_op(s, op_str) {
                let lhs = self.parse_expr(&s[..pos])?;
                let rhs = self.parse_expr(&s[pos + op_str.len()..])?;
                return Ok(Expr::BinOp {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                });
            }
        }

        for &(op_str, op) in &[
            (" & ", BinOp::BitAnd),
            (" | ", BinOp::BitOr),
            (" ^ ", BinOp::BitXor),
            (" << ", BinOp::Shl),
            (" >> ", BinOp::Shr),
        ] {
            if let Some(pos) = find_top_level_op(s, op_str) {
                let lhs = self.parse_expr(&s[..pos])?;
                let rhs = self.parse_expr(&s[pos + op_str.len()..])?;
                return Ok(Expr::BinOp {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                });
            }
        }

        if s.starts_with('(') && s.ends_with(')') {
            return self.parse_expr(&s[1..s.len() - 1]);
        }

        if s.starts_with('-') && s.len() > 1 {
            let inner = self.parse_expr(&s[1..])?;
            return Ok(Expr::UnaryOp {
                op: UnaryOp::Neg,
                operand: Box::new(inner),
            });
        }

        self.parse_atom(s)
    }

    fn parse_atom(&self, s: &str) -> Result<Expr, CompileError> {
        let s = s.trim();

        match s {
            "thread_id()" | "thread_id_x()" => return Ok(Expr::ThreadId(Dimension::X)),
            "thread_id_y()" => return Ok(Expr::ThreadId(Dimension::Y)),
            "thread_id_z()" => return Ok(Expr::ThreadId(Dimension::Z)),
            "workgroup_id()" | "workgroup_id_x()" => return Ok(Expr::WorkgroupId(Dimension::X)),
            "workgroup_size()" | "workgroup_size_x()" => {
                return Ok(Expr::WorkgroupSize(Dimension::X))
            }
            "lane_id()" => return Ok(Expr::LaneId),
            "wave_width()" => return Ok(Expr::WaveWidth),
            "True" | "true" => return Ok(Expr::Literal(Literal::Bool(true))),
            "False" | "false" => return Ok(Expr::Literal(Literal::Bool(false))),
            _ => {}
        }

        if let Some(paren_pos) = s.find('(') {
            if s.ends_with(')') {
                let func_name = &s[..paren_pos];
                let args_str = &s[paren_pos + 1..s.len() - 1];
                return self.parse_call(func_name, args_str);
            }
        }

        if let Some(bracket_pos) = s.find('[') {
            if s.ends_with(']') {
                let base = &s[..bracket_pos];
                let index = &s[bracket_pos + 1..s.len() - 1];
                return Ok(Expr::Index {
                    base: Box::new(self.parse_expr(base)?),
                    index: Box::new(self.parse_expr(index)?),
                });
            }
        }

        if let Ok(v) = s.parse::<i64>() {
            return Ok(Expr::Literal(Literal::Int(v)));
        }
        if let Ok(v) = s.parse::<f64>() {
            return Ok(Expr::Literal(Literal::Float(v)));
        }

        if s.starts_with("0x") || s.starts_with("0X") {
            if let Ok(v) = i64::from_str_radix(&s[2..], 16) {
                return Ok(Expr::Literal(Literal::Int(v)));
            }
        }

        if is_valid_identifier(s) {
            return Ok(Expr::Var(s.to_string()));
        }

        Err(CompileError::ParseError {
            message: format!("cannot parse expression: '{s}'"),
        })
    }

    fn parse_call(&self, func_name: &str, args_str: &str) -> Result<Expr, CompileError> {
        let args: Vec<Expr> = if args_str.trim().is_empty() {
            Vec::new()
        } else {
            args_str
                .split(',')
                .map(|a| self.parse_expr(a.trim()))
                .collect::<Result<_, _>>()?
        };

        match func_name {
            "sqrt" => Ok(Expr::Call {
                func: BuiltinFunc::Sqrt,
                args,
            }),
            "sin" => Ok(Expr::Call {
                func: BuiltinFunc::Sin,
                args,
            }),
            "cos" => Ok(Expr::Call {
                func: BuiltinFunc::Cos,
                args,
            }),
            "exp2" => Ok(Expr::Call {
                func: BuiltinFunc::Exp2,
                args,
            }),
            "log2" => Ok(Expr::Call {
                func: BuiltinFunc::Log2,
                args,
            }),
            "abs" => Ok(Expr::Call {
                func: BuiltinFunc::Abs,
                args,
            }),
            "min" => Ok(Expr::Call {
                func: BuiltinFunc::Min,
                args,
            }),
            "max" => Ok(Expr::Call {
                func: BuiltinFunc::Max,
                args,
            }),
            "atomic_add" => Ok(Expr::Call {
                func: BuiltinFunc::AtomicAdd,
                args,
            }),
            "thread_id" => Ok(Expr::ThreadId(Dimension::X)),
            "workgroup_id" => Ok(Expr::WorkgroupId(Dimension::X)),
            "workgroup_size" => Ok(Expr::WorkgroupSize(Dimension::X)),
            "lane_id" => Ok(Expr::LaneId),
            "wave_width" => Ok(Expr::WaveWidth),
            "int" | "u32" => {
                if args.len() == 1 {
                    Ok(Expr::Cast {
                        expr: Box::new(args.into_iter().next().unwrap()),
                        to: Type::U32,
                    })
                } else {
                    Err(CompileError::ParseError {
                        message: "int() takes 1 argument".to_string(),
                    })
                }
            }
            "float" | "f32" => {
                if args.len() == 1 {
                    Ok(Expr::Cast {
                        expr: Box::new(args.into_iter().next().unwrap()),
                        to: Type::F32,
                    })
                } else {
                    Err(CompileError::ParseError {
                        message: "float() takes 1 argument".to_string(),
                    })
                }
            }
            _ => Err(CompileError::ParseError {
                message: format!("unknown function: {func_name}"),
            }),
        }
    }
}

fn find_top_level_op(s: &str, op: &str) -> Option<usize> {
    let mut depth = 0i32;
    let bytes = s.as_bytes();
    let op_bytes = op.as_bytes();
    let op_len = op.len();

    if s.len() < op_len {
        return None;
    }

    let mut i = s.len() - op_len;
    loop {
        let ch = bytes[i + op_len - 1];
        match ch {
            b')' | b']' => depth += 1,
            b'(' | b'[' => depth -= 1,
            _ => {}
        }
        if depth == 0 && &bytes[i..i + op_len] == op_bytes {
            return Some(i);
        }
        if i == 0 {
            break;
        }
        i -= 1;
    }
    None
}

fn is_valid_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut chars = s.chars();
    let first = chars.next().unwrap();
    if !first.is_alphabetic() && first != '_' {
        return false;
    }
    chars.all(|c| c.is_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_vector_add() {
        let source = r#"
from wave import kernel, f32, thread_id

@kernel
def vector_add(a: f32[:], b: f32[:], out: f32[:], n: u32):
    gid = thread_id()
    if gid < n:
        out[gid] = a[gid] + b[gid]
"#;
        let kernel = parse_python(source).unwrap();
        assert_eq!(kernel.name, "vector_add");
        assert_eq!(kernel.params.len(), 4);
        assert_eq!(kernel.params[0].name, "a");
        assert_eq!(kernel.params[0].ty, Type::Ptr(AddressSpace::Device));
        assert_eq!(kernel.params[3].name, "n");
        assert_eq!(kernel.params[3].ty, Type::U32);
        assert_eq!(kernel.body.len(), 2);
    }

    #[test]
    fn test_parse_simple_assign() {
        let source = r#"
@kernel
def test(n: u32):
    x = 42
    y = x + 1
"#;
        let kernel = parse_python(source).unwrap();
        assert_eq!(kernel.name, "test");
        assert_eq!(kernel.body.len(), 2);
    }

    #[test]
    fn test_parse_expressions() {
        let parser = PythonParser::new(&[]);
        let expr = parser.parse_expr("a + b * c").unwrap();
        match &expr {
            Expr::BinOp { op: BinOp::Add, .. } => {}
            _ => panic!("expected Add at top level"),
        }
    }

    #[test]
    fn test_parse_array_index() {
        let parser = PythonParser::new(&[]);
        let expr = parser.parse_expr("a[i]").unwrap();
        match &expr {
            Expr::Index { base, index } => {
                assert_eq!(**base, Expr::Var("a".into()));
                assert_eq!(**index, Expr::Var("i".into()));
            }
            _ => panic!("expected Index"),
        }
    }

    #[test]
    fn test_parse_thread_id() {
        let parser = PythonParser::new(&[]);
        let expr = parser.parse_expr("thread_id()").unwrap();
        assert_eq!(expr, Expr::ThreadId(Dimension::X));
    }

    #[test]
    fn test_parse_literal() {
        let parser = PythonParser::new(&[]);
        assert_eq!(
            parser.parse_expr("42").unwrap(),
            Expr::Literal(Literal::Int(42))
        );
        assert_eq!(
            parser.parse_expr("3.14").unwrap(),
            Expr::Literal(Literal::Float(3.14))
        );
    }
}
