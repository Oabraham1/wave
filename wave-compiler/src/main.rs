// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! CLI for the WAVE compiler.
//!
//! Compiles GPU kernel source files to WAVE binary (.wbin) format.

use clap::Parser;
use std::fs;
use std::process;

use wave_compiler::driver::config::{CompilerConfig, Language, OptLevel};
use wave_compiler::driver::pipeline::compile_source;

#[derive(Parser)]
#[command(name = "wave-compiler")]
#[command(about = "Compile GPU kernel source to WAVE binary")]
struct Cli {
    /// Input source file.
    input: String,

    /// Output binary file.
    #[arg(short, long)]
    output: String,

    /// Source language (python, rust, cpp, typescript).
    #[arg(long)]
    lang: String,

    /// Optimization level (0, 1, 2, 3).
    #[arg(short = 'O', long, default_value = "0")]
    opt_level: String,

    /// Maximum registers.
    #[arg(long, default_value = "256")]
    max_registers: u32,

    /// Wave width (SIMD lanes).
    #[arg(long, default_value = "32")]
    wave_width: u32,

    /// Dump HIR to stderr.
    #[arg(long)]
    dump_hir: bool,

    /// Dump MIR to stderr.
    #[arg(long)]
    dump_mir: bool,

    /// Dump LIR to stderr.
    #[arg(long)]
    dump_lir: bool,
}

fn main() {
    let cli = Cli::parse();

    let language = Language::from_str_opt(&cli.lang).unwrap_or_else(|| {
        eprintln!("error: unknown language '{}'", cli.lang);
        process::exit(1);
    });

    let opt_level = OptLevel::from_str_opt(&cli.opt_level).unwrap_or_else(|| {
        eprintln!("error: unknown optimization level '{}'", cli.opt_level);
        process::exit(1);
    });

    let config = CompilerConfig {
        language,
        opt_level,
        max_registers: cli.max_registers,
        wave_width: cli.wave_width,
        dump_hir: cli.dump_hir,
        dump_mir: cli.dump_mir,
        dump_lir: cli.dump_lir,
    };

    let source = fs::read_to_string(&cli.input).unwrap_or_else(|e| {
        eprintln!("error: cannot read '{}': {}", cli.input, e);
        process::exit(1);
    });

    match compile_source(&source, &config) {
        Ok(binary) => {
            fs::write(&cli.output, &binary).unwrap_or_else(|e| {
                eprintln!("error: cannot write '{}': {}", cli.output, e);
                process::exit(1);
            });
            eprintln!(
                "compiled {} -> {} ({} bytes)",
                cli.input,
                cli.output,
                binary.len()
            );
        }
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_cli_parse_help() {
        use clap::CommandFactory;
        let cmd = super::Cli::command();
        assert!(cmd.get_about().is_some());
    }

    #[test]
    fn test_language_detection() {
        use wave_compiler::driver::config::Language;
        assert_eq!(Language::from_str_opt("python"), Some(Language::Python));
        assert_eq!(Language::from_str_opt("rs"), Some(Language::Rust));
    }
}
