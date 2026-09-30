// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! CLI for the WAVE compiler. Compiles GPU kernel source files to WBIN format.

use clap::Parser;
use std::fs;

use wave_compiler::driver::config::{CompilerConfig, Language, OptLevel};
use wave_compiler::driver::pipeline::compile_source;

#[derive(Debug, Parser)]
#[command(name = "wave-compiler")]
#[command(about = "Compile GPU kernel source to WAVE binary")]
struct Cli {
    /// Input source file.
    input: String,

    /// Output binary file.
    #[arg(short, long)]
    output: String,

    /// Source language, python, rust, cpp, typescript.
    #[arg(long)]
    lang: String,

    /// Optimization level, 0, 1, 2, 3.
    #[arg(short = 'O', long, default_value = "0")]
    opt_level: String,

    /// Maximum registers.
    #[arg(long, default_value = "256")]
    max_registers: u32,

    /// Wave width, SIMD lanes.
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let language = Language::from_str_opt(&cli.lang)
        .ok_or_else(|| format!("unknown language '{}'", cli.lang))?;
    let opt_level = OptLevel::from_str_opt(&cli.opt_level)
        .ok_or_else(|| format!("unknown optimization level '{}'", cli.opt_level))?;
    let config = CompilerConfig {
        language,
        opt_level,
        max_registers: cli.max_registers,
        wave_width: cli.wave_width,
        dump_hir: cli.dump_hir,
        dump_mir: cli.dump_mir,
        dump_lir: cli.dump_lir,
    };
    let source = fs::read_to_string(&cli.input)
        .map_err(|error| format!("cannot read '{}': {error}", cli.input))?;
    let binary = compile_source(&source, &config)?;
    fs::write(&cli.output, &binary)
        .map_err(|error| format!("cannot write '{}': {error}", cli.output))?;
    eprintln!(
        "compiled {} -> {}: {} bytes",
        cli.input,
        cli.output,
        binary.len()
    );
    Ok(())
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
