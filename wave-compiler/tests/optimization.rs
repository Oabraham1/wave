// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Check observable integer results and memory writes across optimization passes.

use std::collections::{HashMap, HashSet};
use wave_compiler::hir::{expr::BinOp, types::AddressSpace};
use wave_compiler::mir::{
    basic_block::{BasicBlock, PhiNode, Terminator},
    function::MirFunction,
    instruction::{ConstValue, MirInst},
    types::MirType,
    value::{BlockId, ValueId},
};
use wave_compiler::optimize::{
    self, cse::Cse, dce::Dce, licm::Licm, loop_unroll::LoopUnroll, mem2reg::Mem2Reg, pass::Pass,
    strength_reduce::StrengthReduce,
};
use wave_compiler::OptLevel;

fn constant(dest: u32, value: i32) -> MirInst {
    MirInst::Const {
        dest: ValueId(dest),
        value: ConstValue::I32(value),
    }
}

fn binary(dest: u32, op: BinOp, lhs: u32, rhs: u32) -> MirInst {
    MirInst::BinOp {
        dest: ValueId(dest),
        op,
        lhs: ValueId(lhs),
        rhs: ValueId(rhs),
        ty: MirType::I32,
    }
}

fn store(addr: u32, value: u32) -> MirInst {
    MirInst::Store {
        addr: ValueId(addr),
        value: ValueId(value),
        space: AddressSpace::Private,
    }
}

fn function(instructions: Vec<MirInst>) -> MirFunction {
    let mut function = MirFunction::new("integer_results".into(), BlockId(0));
    let mut block = BasicBlock::new(BlockId(0));
    block.instructions = instructions;
    function.blocks.push(block);
    function
}

fn execute(function: &MirFunction) -> Vec<(i32, i32)> {
    let mut values = HashMap::new();
    let mut memory = HashMap::new();
    let mut writes = Vec::new();
    let mut current = function.entry;
    let mut previous = current;
    for _ in 0..1000 {
        let block = function.block(current).expect("branch target exists");
        let incoming: Vec<_> = block
            .phis
            .iter()
            .map(|phi| {
                let source = phi
                    .incoming
                    .iter()
                    .find(|(bid, _)| *bid == previous)
                    .unwrap()
                    .1;
                (phi.dest, values[&source])
            })
            .collect();
        values.extend(incoming);
        for inst in &block.instructions {
            match inst {
                MirInst::Const { dest, value } => {
                    values.insert(*dest, value.to_bits().cast_signed());
                }
                MirInst::BinOp {
                    dest, op, lhs, rhs, ..
                } => {
                    let lhs: i32 = values[lhs];
                    let rhs: i32 = values[rhs];
                    let result = match op {
                        BinOp::Add => lhs.wrapping_add(rhs),
                        BinOp::Mul => lhs.wrapping_mul(rhs),
                        BinOp::Div => lhs.wrapping_div(rhs),
                        BinOp::Mod => lhs.wrapping_rem(rhs),
                        BinOp::Shl => lhs.wrapping_shl(rhs.cast_unsigned()),
                        BinOp::Shr => lhs
                            .cast_unsigned()
                            .wrapping_shr(rhs.cast_unsigned())
                            .cast_signed(),
                        BinOp::Lt => i32::from(lhs < rhs),
                        _ => panic!("unsupported test operation {op:?}"),
                    };
                    values.insert(*dest, result);
                }
                MirInst::Store { addr, value, .. } => {
                    memory.insert(values[addr], values[value]);
                    writes.push((values[addr], values[value]));
                }
                MirInst::Load { dest, addr, .. } => {
                    values.insert(*dest, memory[&values[addr]]);
                }
                _ => panic!("unsupported test instruction {inst:?}"),
            }
        }
        previous = current;
        current = match block.terminator {
            Terminator::Branch { target } => target,
            Terminator::CondBranch {
                cond,
                true_target,
                false_target,
            } => {
                if values[&cond] == 0 {
                    false_target
                } else {
                    true_target
                }
            }
            Terminator::Return => return writes,
        };
    }
    panic!("test program exceeded its instruction bound");
}

fn assert_ssa(function: &MirFunction) {
    let mut definitions = HashSet::new();
    for block in &function.blocks {
        for value in block.defs() {
            assert!(definitions.insert(value), "duplicate definition {value}");
        }
    }
}

#[test]
fn cse_preserves_stored_results() {
    let mut function = function(vec![
        constant(0, 8),
        constant(1, 7),
        constant(2, 32),
        binary(3, BinOp::Add, 0, 1),
        binary(4, BinOp::Add, 0, 1),
        store(2, 4),
    ]);
    assert!(Cse.run(&mut function));
    assert_eq!(execute(&function), vec![(32, 15)]);
    assert_ssa(&function);
}

#[test]
fn cse_keeps_expressions_on_separate_branches() {
    let mut function = function(vec![
        constant(0, 8),
        constant(1, 7),
        constant(2, 32),
        constant(3, 0),
    ]);
    function.blocks[0].terminator = Terminator::CondBranch {
        cond: ValueId(3),
        true_target: BlockId(1),
        false_target: BlockId(2),
    };
    for (bid, dest) in [(1, 4), (2, 5)] {
        let mut block = BasicBlock::new(BlockId(bid));
        block.instructions = vec![binary(dest, BinOp::Add, 0, 1), store(2, dest)];
        function.blocks.push(block);
    }
    assert!(!Cse.run(&mut function));
    assert_eq!(execute(&function), vec![(32, 15)]);
}

#[test]
fn private_loads_forward_values_and_respect_aliasing() {
    let mut function = function(vec![
        constant(0, 32),
        constant(1, 42),
        constant(2, 32),
        constant(3, 99),
        store(0, 1),
        MirInst::Load {
            dest: ValueId(4),
            addr: ValueId(0),
            space: AddressSpace::Private,
            ty: MirType::I32,
        },
        store(2, 3),
        MirInst::Load {
            dest: ValueId(5),
            addr: ValueId(0),
            space: AddressSpace::Private,
            ty: MirType::I32,
        },
        store(0, 4),
        store(0, 5),
    ]);
    function.set_type(ValueId(1), MirType::I32);
    let expected = execute(&function);
    assert!(Mem2Reg.run(&mut function));
    assert_eq!(execute(&function), expected);
    assert_eq!(expected, vec![(32, 42), (32, 99), (32, 42), (32, 99)]);
}

#[test]
fn dce_removes_whole_unused_chains_and_phi_cycles() {
    let mut instructions = vec![constant(0, 1), constant(1, 32)];
    for value in 2..100 {
        instructions.push(binary(value, BinOp::Add, value - 1, 0));
    }
    instructions.push(store(1, 0));
    let mut function = function(instructions);
    function.blocks[0].phis.push(PhiNode {
        dest: ValueId(100),
        ty: MirType::I32,
        incoming: vec![(BlockId(0), ValueId(101))],
    });
    function.blocks[0].phis.push(PhiNode {
        dest: ValueId(101),
        ty: MirType::I32,
        incoming: vec![(BlockId(0), ValueId(100))],
    });
    assert!(Dce.run(&mut function));
    assert_eq!(function.blocks[0].instructions.len(), 3);
    assert_eq!(function.blocks[0].phis, [] as [PhiNode; 0]);
    assert_eq!(execute(&function), vec![(32, 1)]);
    assert!(!Dce.run(&mut function));
}

fn counted_loop(count: i32) -> MirFunction {
    let mut function = function(vec![
        constant(0, 0),
        constant(1, 1),
        constant(2, count),
        constant(3, 32),
        constant(7, 36),
    ]);
    function.blocks[0].terminator = Terminator::Branch { target: BlockId(1) };
    let mut body = BasicBlock::new(BlockId(1));
    body.phis.push(PhiNode {
        dest: ValueId(4),
        ty: MirType::I32,
        incoming: vec![(BlockId(0), ValueId(0)), (BlockId(1), ValueId(5))],
    });
    body.instructions = vec![
        binary(5, BinOp::Add, 4, 1),
        store(3, 5),
        binary(6, BinOp::Lt, 5, 2),
    ];
    body.terminator = Terminator::CondBranch {
        cond: ValueId(6),
        true_target: BlockId(1),
        false_target: BlockId(2),
    };
    let mut exit = BasicBlock::new(BlockId(2));
    exit.instructions.push(store(7, 5));
    function.blocks.extend([body, exit]);
    function
}

#[test]
fn loop_passes_preserve_iteration_count_and_carried_values() {
    for count in [1, 3, 4, 5, 20] {
        let original = counted_loop(count);
        let expected = execute(&original);
        for level in [OptLevel::O1, OptLevel::O2, OptLevel::O3] {
            let mut function = original.clone();
            optimize::optimize(&mut function, level);
            assert_ssa(&function);
            assert_eq!(
                execute(&function),
                expected,
                "count={count}, level={level:?}"
            );
        }
        let mut function = original;
        assert!(!Licm.run(&mut function));
        assert_eq!(LoopUnroll.run(&mut function), count <= 4);
        assert_ssa(&function);
        assert_eq!(execute(&function), expected);
    }
}

#[test]
fn integer_reduction_preserves_signed_results_and_unique_values() {
    let mut function = function(vec![
        constant(0, -7),
        constant(1, 4),
        constant(2, 32),
        constant(10003, 5),
        binary(3, BinOp::Mul, 0, 1),
        binary(4, BinOp::Div, 0, 1),
        binary(5, BinOp::Mod, 0, 1),
        store(2, 3),
        store(2, 4),
        store(2, 5),
        store(2, 10003),
    ]);
    let expected = execute(&function);
    assert!(StrengthReduce.run(&mut function));
    assert_ssa(&function);
    assert_eq!(execute(&function), expected);
    assert_eq!(expected, vec![(32, -28), (32, -1), (32, -3), (32, 5)]);
}

#[test]
fn unrolling_preserves_multiple_loop_carried_values() {
    let mut function = counted_loop(4);
    function.blocks[1].phis.push(PhiNode {
        dest: ValueId(8),
        ty: MirType::I32,
        incoming: vec![(BlockId(0), ValueId(0)), (BlockId(1), ValueId(9))],
    });
    function.blocks[1]
        .instructions
        .insert(1, binary(9, BinOp::Add, 8, 5));
    function.blocks[2].instructions = vec![store(7, 9)];
    let expected = execute(&function);
    assert!(LoopUnroll.run(&mut function));
    assert_ssa(&function);
    assert_eq!(execute(&function), expected);
    assert_eq!(expected.last(), Some(&(36, 10)));
}

#[test]
fn licm_keeps_reads_inside_a_mutating_loop() {
    let mut function = counted_loop(3);
    function.blocks[0].instructions.push(store(3, 0));
    function.blocks[1].instructions.insert(
        0,
        MirInst::Load {
            dest: ValueId(8),
            addr: ValueId(3),
            space: AddressSpace::Private,
            ty: MirType::I32,
        },
    );
    function.blocks[1].instructions.insert(1, store(7, 8));
    let expected = execute(&function);
    assert!(!Licm.run(&mut function));
    assert_eq!(execute(&function), expected);
}
