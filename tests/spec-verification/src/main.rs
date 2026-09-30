// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Runs the WAVE specification checks.

mod harness;
mod real_programs;
mod section2_execution_model;
mod section3_register_model;
mod section4_memory_model;
mod section5_control_flow;
mod section6_instructions;
mod section6_synchronization;
mod section6_wave_ops;
mod section7_capabilities;
mod section8_encoding;
mod section9_conformance;
mod stress_tests;

use harness::TestResult;
type Section = (&'static str, fn() -> Vec<TestResult>);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("WAVE Specification Verification Report");
    println!("=======================================");
    println!("Spec Version: 0.1");
    println!("Toolchain: wave-asm v0.1, wave-emu v0.1");
    println!("Date: {}", chrono_lite::Utc::now());
    println!();

    let mut all_results: Vec<TestResult> = Vec::new();

    let sections: Vec<Section> = vec![
        (
            "Section 2: Execution Model",
            section2_execution_model::run_tests,
        ),
        (
            "Section 3: Register Model",
            section3_register_model::run_tests,
        ),
        ("Section 4: Memory Model", section4_memory_model::run_tests),
        ("Section 5: Control Flow", section5_control_flow::run_tests),
        ("Section 6: Instructions", section6_instructions::run_tests),
        ("Section 6: Wave Operations", section6_wave_ops::run_tests),
        (
            "Section 6: Synchronization",
            section6_synchronization::run_tests,
        ),
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
            println!("  [{status}] {}: {}", result.name, result.spec_claim);
            println!("    {} instructions", result.cycles);
            if !result.passed {
                println!("    {}", result.details);
            }
        }
        println!();

        all_results.extend(results);
    }

    let failed = all_results.iter().filter(|r| !r.passed).count();

    if failed > 0 {
        println!();
        println!("Failed checks:");
        for (i, result) in all_results.iter().filter(|r| !r.passed).enumerate() {
            println!("  {}. {}: {}", i + 1, result.name, result.spec_section);
            println!("     {}", result.details);
        }
    }

    if failed > 0 {
        return Err(format!("{failed} spec checks failed").into());
    }
    println!("Specification checks passed.");
    Ok(())
}

// Simple date helper since we don't want heavy dependencies
mod chrono_lite {
    #[derive(Debug)]
    pub struct Utc;
    impl Utc {
        pub fn now() -> String {
            std::process::Command::new("date")
                .arg("+%Y-%m-%d")
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .unwrap_or_else(|_| "unknown".to_string())
        }
    }
}
