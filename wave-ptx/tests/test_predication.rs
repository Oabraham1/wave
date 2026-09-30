// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Check predicate capture, converged shuffles, and control-flow validation.

use wave_decode::opcodes::{CmpOp, MemWidth, WaveOpType};
use wave_decode::{DecodedInstruction, KernelInfo, Operation};
use wave_ptx::codegen::CodeGenerator;

fn kernel() -> KernelInfo {
    KernelInfo {
        name: "mask_test".into(),
        register_count: 8,
        local_memory_size: 0,
        workgroup_size: [32, 1, 1],
        code_offset: 0,
        code_size: 0,
    }
}

fn instruction(operation: Operation) -> DecodedInstruction {
    DecodedInstruction {
        offset: 0,
        size: 4,
        operation,
        predicate: 0,
        predicate_negated: false,
    }
}

fn shuffle() -> Operation {
    Operation::WaveOp {
        op: WaveOpType::Shuffle,
        rd: 2,
        rs1: 2,
        rs2: Some(2),
    }
}

#[test]
fn guard_capture_survives_a_write_to_its_predicate() {
    let mut guarded = instruction(Operation::Icmp {
        op: CmpOp::Eq,
        pd: 0,
        rs1: 0,
        rs2: 1,
    });
    guarded.predicate_negated = true;
    let mut generator = CodeGenerator::new();
    let output = generator.generate(&[guarded], &kernel()).unwrap();
    let capture = output.find("not.pred %wave_condition_0, %p0;").unwrap();
    let write = output
        .find("@%wave_active setp.eq.s32 %p0, %r0, %r1;")
        .unwrap();
    let restore = output
        .find("and.pred %wave_active, %wave_entry_0, %wave_entry_0;")
        .unwrap();
    assert!(capture < write && write < restore);
}

#[test]
fn guarded_shuffle_in_a_branch_keeps_all_lanes_participating() {
    let mut guarded = instruction(shuffle());
    guarded.predicate = 2;
    let code = [
        instruction(Operation::If { ps: 1 }),
        guarded,
        instruction(Operation::Endif),
    ];
    let mut generator = CodeGenerator::new();
    let output = generator.generate(&code, &kernel()).unwrap();
    assert_eq!(output.matches("shfl.sync.idx.b32 ").count(), 1);
    assert!(
        !output.contains("bra ") && !output.contains("bra.uni "),
        "shuffle participation must not branch: {output}"
    );
    let gather = output
        .lines()
        .find(|line| line.contains("shfl.sync"))
        .unwrap();
    assert_eq!(
        gather.trim(),
        "shfl.sync.idx.b32 %wave_value, %wave_source, %wave_index, 31, 0xFFFFFFFF;"
    );
    let snapshot = output.find("mov.b32 %wave_source, %r2;").unwrap();
    let gather = output.find(gather).unwrap();
    let write = output
        .find("@%wave_active mov.b32 %r2, %wave_value;")
        .unwrap();
    assert!(snapshot < gather && gather < write);
    assert!(output.contains("selp.b32 %wave_index, %r2, %wave_lane, %wave_active;"));
    assert!(output.contains("@%wave_active mov.b32 %r2, %wave_value;"));
}

#[test]
fn memory_accesses_preserve_their_order_around_a_shuffle() {
    let code = [
        instruction(Operation::MovImm { rd: 0, imm: 0 }),
        instruction(Operation::DeviceLoad {
            width: MemWidth::U32,
            rd: 2,
            addr: 0,
        }),
        instruction(Operation::MovImm { rd: 2, imm: 0 }),
        instruction(shuffle()),
        instruction(Operation::DeviceStore {
            width: MemWidth::U32,
            addr: 0,
            value: 2,
        }),
        instruction(Operation::Halt),
    ];
    let mut generator = CodeGenerator::new();
    let output = generator.generate(&code, &kernel()).unwrap();
    let load = output.find("ld.global.u32 %r2, [%rd1];").unwrap();
    let write = output.find("@%wave_active mov.b32 %r2, 0;").unwrap();
    let shuffle = output.find("shfl.sync.idx.b32 ").unwrap();
    let store = output.find("st.global.u32 [%rd1], %r2;").unwrap();
    assert!(load < write && write < shuffle && shuffle < store);
    for line in output.lines().map(str::trim) {
        if line.contains("ld.global") || line.contains("st.global") {
            assert!(
                !line.starts_with('@'),
                "memory access was predicated: {line}"
            );
        }
    }
}

#[test]
fn loops_and_interleaved_memory_use_instruction_lowering() {
    let cases = [
        vec![
            instruction(Operation::Loop),
            instruction(shuffle()),
            instruction(Operation::Endloop),
        ],
        vec![
            instruction(shuffle()),
            instruction(Operation::DeviceLoad {
                width: MemWidth::U32,
                rd: 2,
                addr: 0,
            }),
        ],
        vec![
            instruction(Operation::DeviceStore {
                width: MemWidth::U32,
                addr: 0,
                value: 2,
            }),
            instruction(Operation::DeviceLoad {
                width: MemWidth::U32,
                rd: 2,
                addr: 0,
            }),
        ],
    ];
    for code in cases {
        let mut generator = CodeGenerator::new();
        let output = generator.generate(&code, &kernel()).unwrap();
        assert!(!output.contains("%wave_active"), "PTX: {output}");
        for instruction in code {
            let operation = match instruction.operation {
                Operation::Loop => "bra.uni $L_loop_0;",
                Operation::WaveOp { .. } => "shfl.sync.idx.b32 %r2, %r2, %r2, 31, 0xFFFFFFFF;",
                Operation::DeviceLoad { .. } => "ld.global.u32 %r2, [%rd1];",
                Operation::DeviceStore { .. } => "st.global.u32 [%rd1], %r2;",
                _ => continue,
            };
            assert!(output.contains(operation), "missing {operation}: {output}");
        }
    }
}

#[test]
fn malformed_control_is_rejected_and_the_generator_can_be_reused() {
    let mut generator = CodeGenerator::new();
    let expected = generator
        .generate(&[instruction(shuffle())], &kernel())
        .unwrap();
    for code in [
        vec![instruction(Operation::Else)],
        vec![instruction(Operation::Endif)],
        vec![instruction(Operation::If { ps: 1 })],
        vec![
            instruction(Operation::If { ps: 1 }),
            instruction(Operation::Else),
            instruction(Operation::Else),
            instruction(Operation::Endif),
        ],
    ] {
        assert!(generator.generate(&code, &kernel()).is_err());
        assert_eq!(
            generator
                .generate(&[instruction(shuffle())], &kernel())
                .unwrap(),
            expected
        );
    }
}

#[test]
fn fallback_preserves_negated_predicate_zero() {
    let mut operation = instruction(Operation::Fadd {
        rd: 2,
        rs1: 0,
        rs2: 1,
    });
    operation.predicate_negated = true;
    let mut generator = CodeGenerator::new();
    let output = generator.generate(&[operation], &kernel()).unwrap();
    assert!(output.contains("@!%p0 add.rn.f32"));
}

#[test]
fn eligible_registers_and_predicates_start_at_zero() {
    let mut generator = CodeGenerator::new();
    let output = generator
        .generate(&[instruction(shuffle())], &kernel())
        .unwrap();
    let core = output.find("setp.eq.u32 %wave_active, 0, 0;").unwrap();
    for index in 0..kernel().register_count {
        let zero = output.find(&format!("mov.b32 %r{index}, 0;")).unwrap();
        assert!(zero < core);
    }
    for index in 0..4 {
        let zero = output
            .find(&format!("setp.ne.u32 %p{index}, 0, 0;"))
            .unwrap();
        assert!(zero < core);
    }
}

#[test]
fn every_step_of_a_multiline_local_is_guarded() {
    let mut code = vec![instruction(Operation::If { ps: 1 })];
    for sr in [3, 15] {
        code.push(instruction(Operation::MovSr {
            rd: 2,
            sr_index: sr,
        }));
    }
    code.push(instruction(Operation::Shl {
        rd: 2,
        rs1: 1,
        rs2: 2,
    }));
    code.push(instruction(Operation::Endif));
    let mut generator = CodeGenerator::new();
    let output = generator.generate(&code, &kernel()).unwrap();
    let start = output
        .find("and.pred %wave_active, %wave_entry_0, %wave_condition_0;")
        .unwrap();
    let end = output
        .find("and.pred %wave_active, %wave_else_0, %wave_else_0;")
        .unwrap();
    let body: Vec<_> = output[start..end]
        .lines()
        .skip(1)
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    assert!(
        body.len() > 4,
        "special registers must exercise multiple instructions"
    );
    for line in body {
        assert!(
            line.starts_with("@%wave_active "),
            "unguarded local: {line}"
        );
    }
    assert!(output.contains("@%wave_active and.b32 %t0, %r2, 31;"));
    assert!(output.contains("@%wave_active shl.b32 %r2, %r1, %t0;"));
}

#[test]
fn predicated_structural_control_is_rejected() {
    for negated in [false, true] {
        let mut enter = instruction(Operation::If { ps: 1 });
        enter.predicate = u8::from(!negated);
        enter.predicate_negated = negated;
        let mut generator = CodeGenerator::new();
        assert!(generator
            .generate(&[enter, instruction(Operation::Endif)], &kernel())
            .is_err());
    }
}
