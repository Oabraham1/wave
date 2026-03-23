// Copyright (c) 2026 Ojima Abraham. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

// WAVE Specification Verification Test Suite
// Runs comprehensive tests against the WAVE spec to verify correctness,
// completeness, and consistency of the specification.

mod harness;
mod section2_execution_model;
mod section3_register_model;
mod section4_memory_model;
mod section5_control_flow;
mod section6_instructions;
mod section6_wave_ops;
mod section6_synchronization;
mod section7_capabilities;
mod section8_encoding;
mod section9_conformance;
mod real_programs;
mod stress_tests;

use harness::TestResult;
use std::collections::HashMap;

fn main() {
    println!("WAVE Specification Verification Report");
    println!("=======================================");
    println!("Spec Version: 0.1");
    println!("Toolchain: wave-asm v0.1, wave-emu v0.1");
    println!("Date: {}", chrono_lite::Utc::now());
    println!();

    let mut all_results: Vec<TestResult> = Vec::new();
    let mut section_results: HashMap<String, Vec<TestResult>> = HashMap::new();

    // Run all test sections
    let sections: Vec<(&str, fn() -> Vec<TestResult>)> = vec![
        ("Section 2: Execution Model", section2_execution_model::run_tests),
        ("Section 3: Register Model", section3_register_model::run_tests),
        ("Section 4: Memory Model", section4_memory_model::run_tests),
        ("Section 5: Control Flow", section5_control_flow::run_tests),
        ("Section 6: Instructions", section6_instructions::run_tests),
        ("Section 6: Wave Operations", section6_wave_ops::run_tests),
        ("Section 6: Synchronization", section6_synchronization::run_tests),
        ("Section 7: Capabilities", section7_capabilities::run_tests),
        ("Section 8: Binary Encoding", section8_encoding::run_tests),
        ("Section 9: Conformance", section9_conformance::run_tests),
        ("Real Programs", real_programs::run_tests),
        ("Stress Tests", stress_tests::run_tests),
    ];

    for (section_name, run_fn) in sections {
        println!("{}", section_name);
        let results = run_fn();

        for result in &results {
            let status = if result.passed { "PASS" } else { "FAIL" };
            println!("  [{}] {} — {}", status, result.name, result.spec_claim);
            if !result.passed {
                println!("    {}", result.details);
            }
        }
        println!();

        section_results.insert(section_name.to_string(), results.clone());
        all_results.extend(results);
    }

    // Summary
    let passed = all_results.iter().filter(|r| r.passed).count();
    let failed = all_results.iter().filter(|r| !r.passed).count();
    let total = all_results.len();

    println!("Summary: {}/{} PASSED, {} FAILED", passed, total, failed);

    if failed > 0 {
        println!();
        println!("Failed tests indicate spec gaps:");
        for (i, result) in all_results.iter().filter(|r| !r.passed).enumerate() {
            println!("  {}. {} ({})", i + 1, result.name, result.spec_section);
            println!("     {}", result.details);
        }
    }

    std::process::exit(if failed > 0 { 1 } else { 0 });
}

// Simple date helper since we don't want heavy dependencies
mod chrono_lite {
    pub struct Utc;
    impl Utc {
        pub fn now() -> String {
            // Use system date command for simplicity
            std::process::Command::new("date")
                .arg("+%Y-%m-%d")
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .unwrap_or_else(|_| "unknown".to_string())
        }
    }
}
