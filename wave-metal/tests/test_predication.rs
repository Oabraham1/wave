// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Check predicate capture, shuffle participation and control flow.

use wave_decode::opcodes::{CmpOp, MemWidth, WaveOpType};
use wave_decode::{DecodedInstruction, KernelInfo, Operation};
use wave_metal::codegen::CodeGenerator;

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
    let capture = output.find("const bool _wave_condition_0 = !p0;").unwrap();
    let write = output.find("p0 = (int32_t)r0 == (int32_t)r1;").unwrap();
    assert!(capture < write);
    assert!(output.contains("_wave_active = _wave_active && _wave_condition_0;"));
    assert!(!output.contains("_wave_active = _wave_active && !p0;"));
}

#[test]
fn nested_predicated_shuffle_keeps_all_lanes_participating() {
    let mut guarded = instruction(shuffle());
    guarded.predicate = 2;
    let code = [
        instruction(Operation::If { ps: 1 }),
        guarded,
        instruction(Operation::Endif),
    ];
    let mut generator = CodeGenerator::new();
    let output = generator.generate(&code, &kernel()).unwrap();
    assert!(output.contains("const bool _wave_condition_0 = p1;"));
    assert!(output.contains("const bool _wave_condition_1 = p2;"));
    assert_eq!(output.matches("simd_shuffle(").count(), 1);
    assert!(
        !output.contains("if ("),
        "shuffle participation must not branch: {output}"
    );
    assert!(output.contains("const uint32_t _wave_source_1 = r2;"));
    assert!(output.contains("select(lane_id, r2, _wave_active)"));
    assert!(output.contains("r2 = select(r2, _wave_gathered_1, _wave_active);"));
    let inner_restore = output.find("_wave_active = _wave_entry_1;").unwrap();
    let outer_restore = output.find("_wave_active = _wave_entry_0;").unwrap();
    assert!(inner_restore < outer_restore);
}

#[test]
fn predicated_shuffle_preserves_surrounding_memory_operations() {
    let mut guarded = instruction(shuffle());
    guarded.predicate = 1;
    let code = [
        instruction(Operation::MovImm { rd: 0, imm: 0 }),
        instruction(Operation::DeviceLoad {
            width: MemWidth::U32,
            rd: 2,
            addr: 0,
        }),
        instruction(Operation::MovImm { rd: 2, imm: 0 }),
        guarded,
        instruction(Operation::DeviceStore {
            width: MemWidth::U32,
            addr: 0,
            value: 2,
        }),
        instruction(Operation::Halt),
    ];
    let mut generator = CodeGenerator::new();
    let output = generator.generate(&code, &kernel()).unwrap();
    let load = output
        .find("r2 = (uint32_t)(*(device uint32_t*)(device_mem + r0));")
        .unwrap();
    let capture = output.find("const bool _wave_condition_0 = p1;").unwrap();
    let gather = output.find("simd_shuffle(").unwrap();
    let restore = output.find("_wave_active = _wave_entry_0;").unwrap();
    let store = output
        .find("*(device uint32_t*)(device_mem + r0) = (uint32_t)r2;")
        .unwrap();
    assert!(load < capture && capture < gather && gather < restore && restore < store);
}

#[test]
fn loops_and_interleaved_memory_keep_structured_emission() {
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
        assert!(!output.contains("_wave_active"));
        if matches!(code[0].operation, Operation::Loop) {
            assert!(output.contains("while (true) {"));
            assert!(output.contains("r2 = simd_shuffle(r2, r2);"));
        }
    }
}

#[test]
fn malformed_control_is_rejected() {
    let mut generator = CodeGenerator::new();
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
    assert!(output.contains("if (!p0) {"));
}
