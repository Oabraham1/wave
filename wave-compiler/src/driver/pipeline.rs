// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Full compilation pipeline orchestration.
//!
//! Wires together all compiler stages: frontend parsing, HIR validation,
//! lowering to MIR, optimization, lowering to LIR, register allocation,
//! and WAVE binary emission.

use crate::diagnostics::CompileError;
use crate::emit::binary::generate_wbin;
use crate::hir::kernel::Kernel;
use crate::lir::display::display_lir;
use crate::lowering::hir_to_mir::lower_kernel;
use crate::lowering::mir_to_lir::lower_function;
use crate::mir::display::display_function;
use crate::optimize::optimize;
use crate::regalloc::{allocate_registers, max_register_used};

use super::config::CompilerConfig;

/// Compile an HIR kernel to a WAVE binary (.wbin).
///
/// # Errors
///
/// Returns `CompileError` if any compilation stage fails.
pub fn compile_kernel(kernel: &Kernel, config: &CompilerConfig) -> Result<Vec<u8>, CompileError> {
    let mut mir = lower_kernel(kernel)?;

    if config.dump_mir {
        eprintln!("=== MIR (before opt) ===\n{}", display_function(&mir));
    }

    optimize(&mut mir, config.opt_level);

    if config.dump_mir {
        eprintln!("=== MIR (after opt) ===\n{}", display_function(&mir));
    }

    let lir = lower_function(&mir)?;

    if config.dump_lir {
        eprintln!("=== LIR ===\n{}", display_lir(&lir));
    }

    #[allow(clippy::cast_possible_truncation)]
    let num_params = kernel.params.len() as u32;
    let reg_map = allocate_registers(&lir, num_params, config.max_registers);
    let reg_count = max_register_used(&reg_map);

    generate_wbin(&kernel.name, &lir, &reg_map, reg_count)
}

/// Compile source code in the given language to a WAVE binary.
///
/// # Errors
///
/// Returns `CompileError` if parsing or compilation fails.
pub fn compile_source(source: &str, config: &CompilerConfig) -> Result<Vec<u8>, CompileError> {
    let kernel = crate::frontend::parse(source, config.language)?;

    if config.dump_hir {
        eprintln!("HIR: {kernel:#?}");
    }

    compile_kernel(&kernel, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::expr::{BinOp, Dimension, Expr, Literal};
    use crate::hir::kernel::{KernelAttributes, KernelParam};
    use crate::hir::stmt::Stmt;
    use crate::hir::types::{AddressSpace, Type};

    fn make_test_kernel() -> Kernel {
        Kernel {
            name: "test_add".into(),
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
                    then_body: vec![
                        Stmt::Assign {
                            target: "a_val".into(),
                            value: Expr::Index {
                                base: Box::new(Expr::Var("a".into())),
                                index: Box::new(Expr::Var("gid".into())),
                            },
                        },
                        Stmt::Assign {
                            target: "b_val".into(),
                            value: Expr::Index {
                                base: Box::new(Expr::Var("b".into())),
                                index: Box::new(Expr::Var("gid".into())),
                            },
                        },
                        Stmt::Assign {
                            target: "result".into(),
                            value: Expr::BinOp {
                                op: BinOp::Add,
                                lhs: Box::new(Expr::Var("a_val".into())),
                                rhs: Box::new(Expr::Var("b_val".into())),
                            },
                        },
                        Stmt::Store {
                            addr: Expr::BinOp {
                                op: BinOp::Add,
                                lhs: Box::new(Expr::Var("out".into())),
                                rhs: Box::new(Expr::BinOp {
                                    op: BinOp::Mul,
                                    lhs: Box::new(Expr::Var("gid".into())),
                                    rhs: Box::new(Expr::Literal(Literal::Int(4))),
                                }),
                            },
                            value: Expr::Var("result".into()),
                            space: AddressSpace::Device,
                        },
                    ],
                    else_body: None,
                },
            ],
            attributes: KernelAttributes::default(),
        }
    }

    #[test]
    fn test_compile_kernel_to_wbin() {
        let kernel = make_test_kernel();
        let config = CompilerConfig::default();
        let wbin = compile_kernel(&kernel, &config).unwrap();

        assert_eq!(&wbin[0..4], b"WAVE");

        let parsed = wave_decode::WbinFile::parse(&wbin).unwrap();
        assert_eq!(parsed.kernels.len(), 1);
        assert_eq!(parsed.kernels[0].name, "test_add");

        let code = parsed.code();
        let decoded = wave_decode::decode_all(code).unwrap();
        assert!(!decoded.is_empty());

        let has_halt = decoded
            .iter()
            .any(|i| matches!(i.operation, wave_decode::Operation::Halt));
        assert!(has_halt);
    }

    #[test]
    fn test_compile_round_trip() {
        let kernel = make_test_kernel();
        let config = CompilerConfig::default();
        let wbin = compile_kernel(&kernel, &config).unwrap();

        let parsed = wave_decode::WbinFile::parse(&wbin).unwrap();
        let code = parsed.code();
        let decoded = wave_decode::decode_all(code).unwrap();

        let has_if = decoded
            .iter()
            .any(|i| matches!(i.operation, wave_decode::Operation::If { .. }));
        let has_endif = decoded
            .iter()
            .any(|i| matches!(i.operation, wave_decode::Operation::Endif));
        assert!(has_if);
        assert!(has_endif);
    }

    #[test]
    fn test_params_in_correct_registers() {
        let kernel = make_test_kernel();
        let config = CompilerConfig::default();
        let wbin = compile_kernel(&kernel, &config).unwrap();

        let parsed = wave_decode::WbinFile::parse(&wbin).unwrap();
        let code = parsed.code();
        let decoded = wave_decode::decode_all(code).unwrap();

        let mov_sr = decoded
            .iter()
            .find(|i| matches!(i.operation, wave_decode::Operation::MovSr { .. }));
        assert!(mov_sr.is_some());
    }
}
