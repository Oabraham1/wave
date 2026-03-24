// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! CLI entry point. Parses arguments via clap, reads source file, invokes the
//!
//! assembler library, handles diagnostics output, and writes the binary result.
//! Supports verbose mode, AST dumping, hex output, and symbol stripping.

#![allow(clippy::struct_excessive_bools)]

use clap::{Parser, ValueEnum};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use wave_asm::diagnostics::DiagnosticEmitter;
use wave_asm::lexer::Lexer;
use wave_asm::parser::Parser as WaveParser;

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum WarnLevel {
    All,
    None,
    Error,
}

#[derive(Parser)]
#[command(name = "wave-asm")]
#[command(about = "WAVE assembler - translates .wave assembly to .wbin binary")]
#[command(version)]
struct Args {
    #[arg(help = "Input .wave assembly file")]
    input: PathBuf,

    #[arg(short, long, help = "Output .wbin file (default: input with .wbin extension)")]
    output: Option<PathBuf>,

    #[arg(long, help = "Print encoded instructions to stdout")]
    verbose: bool,

    #[arg(long, help = "Dump the AST to stdout")]
    dump_ast: bool,

    #[arg(long, help = "Dump hex output of binary to stdout")]
    dump_hex: bool,

    #[arg(long, help = "Strip symbol table from output")]
    no_symbols: bool,

    #[arg(short = 'W', long = "warn", value_enum, default_value = "all", help = "Warning level")]
    warn_level: WarnLevel,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let source = match fs::read_to_string(&args.input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: failed to read {}: {}", args.input.display(), e);
            return ExitCode::FAILURE;
        }
    };

    let source_name = args.input.to_string_lossy();

    if args.dump_ast {
        let tokens = match Lexer::new(&source).tokenize() {
            Ok(t) => t,
            Err(e) => {
                let emitter = DiagnosticEmitter::new(&source_name, &source);
                emitter.emit_error(&e, &mut std::io::stderr());
                return ExitCode::FAILURE;
            }
        };

        let program = match WaveParser::new(tokens).parse() {
            Ok(p) => p,
            Err(e) => {
                let emitter = DiagnosticEmitter::new(&source_name, &source);
                emitter.emit_error(&e, &mut std::io::stderr());
                return ExitCode::FAILURE;
            }
        };

        println!("{program:#?}");
        return ExitCode::SUCCESS;
    }

    let options = wave_asm::AssemblerOptions {
        strip_symbols: args.no_symbols,
    };

    match wave_asm::assemble_with_options(&source, &source_name, &options) {
        Ok(result) => {
            let emitter = DiagnosticEmitter::new(&source_name, &source);
            let mut stderr = std::io::stderr();

            if args.warn_level == WarnLevel::Error && !result.warnings.is_empty() {
                for warning in &result.warnings {
                    emitter.emit_warning(warning, &mut stderr);
                }
                eprintln!("error: warnings treated as errors (-Werror)");
                return ExitCode::FAILURE;
            }

            if args.warn_level == WarnLevel::All {
                for warning in &result.warnings {
                    emitter.emit_warning(warning, &mut stderr);
                }
            }

            if args.dump_hex {
                for (i, chunk) in result.binary.chunks(16).enumerate() {
                    print!("{:08x}: ", i * 16);
                    for byte in chunk {
                        print!("{byte:02x} ");
                    }
                    println!();
                }
                return ExitCode::SUCCESS;
            }

            let output_path = args.output.unwrap_or_else(|| args.input.with_extension("wbin"));

            if let Err(e) = fs::write(&output_path, &result.binary) {
                eprintln!("error: failed to write {}: {}", output_path.display(), e);
                return ExitCode::FAILURE;
            }

            if args.verbose {
                println!(
                    "Assembled {} bytes to {}",
                    result.binary.len(),
                    output_path.display()
                );
            }

            ExitCode::SUCCESS
        }
        Err(e) => {
            let emitter = DiagnosticEmitter::new(&source_name, &source);
            emitter.emit_error(&e, &mut std::io::stderr());
            ExitCode::FAILURE
        }
    }
}
