// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! HIR expression types for WAVE GPU kernels.
//!
//! Expressions represent computations that produce values, including
//! arithmetic, memory access, GPU intrinsics, and type conversions.

use super::types::{AddressSpace, Type};

/// Dimension index for multi-dimensional GPU dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dimension {
    /// X dimension (index 0).
    X,
    /// Y dimension (index 1).
    Y,
    /// Z dimension (index 2).
    Z,
}

/// Shuffle modes for wave-level data exchange.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShuffleMode {
    /// Direct shuffle to a specific lane.
    Direct,
    /// Shuffle up by an offset.
    Up,
    /// Shuffle down by an offset.
    Down,
    /// Shuffle with XOR of lane IDs.
    Xor,
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
    /// Floor division.
    FloorDiv,
    /// Modulo.
    Mod,
    /// Power.
    Pow,
    /// Bitwise AND.
    BitAnd,
    /// Bitwise OR.
    BitOr,
    /// Bitwise XOR.
    BitXor,
    /// Left shift.
    Shl,
    /// Right shift.
    Shr,
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// Arithmetic negation.
    Neg,
    /// Bitwise NOT.
    BitNot,
    /// Logical NOT.
    Not,
}

/// Built-in math and GPU functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinFunc {
    /// Square root.
    Sqrt,
    /// Sine.
    Sin,
    /// Cosine.
    Cos,
    /// Base-2 exponential.
    Exp2,
    /// Base-2 logarithm.
    Log2,
    /// Absolute value.
    Abs,
    /// Minimum of two values.
    Min,
    /// Maximum of two values.
    Max,
    /// Atomic add.
    AtomicAdd,
}

/// Literal values.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    /// Integer literal.
    Int(i64),
    /// Unsigned integer literal.
    UInt(u64),
    /// Float literal.
    Float(f64),
    /// Boolean literal.
    Bool(bool),
}

/// Memory scope for fence operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryScope {
    /// Wave-level scope.
    Wave,
    /// Workgroup-level scope.
    Workgroup,
    /// Device-level scope.
    Device,
}

/// HIR expression producing a value.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Variable reference.
    Var(String),
    /// Literal constant.
    Literal(Literal),
    /// Binary operation.
    BinOp {
        /// The operator.
        op: BinOp,
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },
    /// Unary operation.
    UnaryOp {
        /// The operator.
        op: UnaryOp,
        /// Operand.
        operand: Box<Expr>,
    },
    /// Built-in function call.
    Call {
        /// The function being called.
        func: BuiltinFunc,
        /// Arguments to the function.
        args: Vec<Expr>,
    },
    /// Array/buffer indexing.
    Index {
        /// Base pointer/array.
        base: Box<Expr>,
        /// Index expression.
        index: Box<Expr>,
    },
    /// Type cast.
    Cast {
        /// Expression to cast.
        expr: Box<Expr>,
        /// Target type.
        to: Type,
    },
    /// Thread ID in a given dimension.
    ThreadId(Dimension),
    /// Workgroup ID in a given dimension.
    WorkgroupId(Dimension),
    /// Workgroup size in a given dimension.
    WorkgroupSize(Dimension),
    /// Lane ID within wave.
    LaneId,
    /// Wave width (number of lanes).
    WaveWidth,
    /// Load from memory.
    Load {
        /// Address to load from.
        addr: Box<Expr>,
        /// Address space.
        space: AddressSpace,
    },
    /// Wave shuffle operation.
    Shuffle {
        /// Value to shuffle.
        value: Box<Expr>,
        /// Target lane or offset.
        lane: Box<Expr>,
        /// Shuffle mode.
        mode: ShuffleMode,
    },
}

impl BinOp {
    /// Returns true if this operator produces a boolean result.
    #[must_use]
    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            Self::Eq | Self::Ne | Self::Lt | Self::Le | Self::Gt | Self::Ge
        )
    }

    /// Returns true if this operator is arithmetic.
    #[must_use]
    pub fn is_arithmetic(&self) -> bool {
        matches!(
            self,
            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::FloorDiv | Self::Mod | Self::Pow
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binop_classification() {
        assert!(BinOp::Eq.is_comparison());
        assert!(BinOp::Lt.is_comparison());
        assert!(!BinOp::Add.is_comparison());
        assert!(BinOp::Add.is_arithmetic());
        assert!(BinOp::Mul.is_arithmetic());
        assert!(!BinOp::Eq.is_arithmetic());
    }

    #[test]
    fn test_expr_construction() {
        let add_expr = Expr::BinOp {
            op: BinOp::Add,
            lhs: Box::new(Expr::Var("a".into())),
            rhs: Box::new(Expr::Var("b".into())),
        };
        match &add_expr {
            Expr::BinOp { op, lhs, rhs } => {
                assert_eq!(*op, BinOp::Add);
                assert_eq!(*lhs, Box::new(Expr::Var("a".into())));
                assert_eq!(*rhs, Box::new(Expr::Var("b".into())));
            }
            _ => panic!("expected BinOp"),
        }
    }

    #[test]
    fn test_index_expr() {
        let idx = Expr::Index {
            base: Box::new(Expr::Var("arr".into())),
            index: Box::new(Expr::ThreadId(Dimension::X)),
        };
        match &idx {
            Expr::Index { base, index } => {
                assert_eq!(**base, Expr::Var("arr".into()));
                assert_eq!(**index, Expr::ThreadId(Dimension::X));
            }
            _ => panic!("expected Index"),
        }
    }

    #[test]
    fn test_literal_variants() {
        let int_lit = Literal::Int(42);
        let float_lit = Literal::Float(3.14);
        let bool_lit = Literal::Bool(true);
        assert_eq!(int_lit, Literal::Int(42));
        assert_eq!(float_lit, Literal::Float(3.14));
        assert_eq!(bool_lit, Literal::Bool(true));
    }
}
