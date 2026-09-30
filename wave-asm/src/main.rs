// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Assemble WAVE source from the command line and report diagnostics.

use clap::{Parser, ValueEnum};
use std::fs;
use std::path::PathBuf;
use wave_asm::diagnostics::DiagnosticEmitter;
use wave_asm::lexer::Lexer;
use wave_asm::parser::Parser as WaveParser;

#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
enum WarnLevel {
    All,
    None,
    Error,
}

#[derive(Debug, Parser)]
#[command(name = "wave-asm")]
#[command(about = "WAVE assembler - translates .wave assembly to .wbin binary")]
#[command(version)]
struct Args {
    #[arg(help = "Input .wave assembly file")]
    input: PathBuf,

    #[arg(
        short,
        long,
        help = "Output file. Defaults to the input path with a .wbin extension"
    )]
    output: Option<PathBuf>,

    #[command(flatten)]
    display: DisplayOptions,

    #[arg(long, help = "Strip symbol table from output")]
    no_symbols: bool,

    #[arg(
        short = 'W',
        long = "warn",
        value_enum,
        default_value = "all",
        help = "Warning level"
    )]
    warn_level: WarnLevel,
}

#[derive(Debug, clap::Args)]
struct DisplayOptions {
    #[arg(long, help = "Print encoded instructions to stdout")]
    verbose: bool,

    #[arg(long, help = "Dump the AST to stdout")]
    dump_ast: bool,

    #[arg(long, help = "Dump hex output of binary to stdout")]
    dump_hex: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let source = match fs::read_to_string(&args.input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: failed to read {}: {}", args.input.display(), e);
            return Err("assembly failed".into());
        }
    };

    let source_name = args.input.to_string_lossy();

    if args.display.dump_ast {
        let tokens = match Lexer::new(&source).tokenize() {
            Ok(t) => t,
            Err(e) => {
                let emitter = DiagnosticEmitter::new(&source_name, &source);
                emitter.emit_error(&e, &mut std::io::stderr());
                return Err("assembly failed".into());
            }
        };

        let program = match WaveParser::new(tokens).parse() {
            Ok(p) => p,
            Err(e) => {
                let emitter = DiagnosticEmitter::new(&source_name, &source);
                emitter.emit_error(&e, &mut std::io::stderr());
                return Err("assembly failed".into());
            }
        };

        println!("{program:#?}");
        return Ok(());
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
                eprintln!("error: warnings treated as errors by -Werror");
                return Err("assembly failed".into());
            }

            if args.warn_level == WarnLevel::All {
                for warning in &result.warnings {
                    emitter.emit_warning(warning, &mut stderr);
                }
            }

            if args.display.dump_hex {
                for (i, chunk) in result.binary.chunks(16).enumerate() {
                    print!("{:08x}: ", i * 16);
                    for byte in chunk {
                        print!("{byte:02x} ");
                    }
                    println!();
                }
                return Ok(());
            }

            let output_path = args
                .output
                .unwrap_or_else(|| args.input.with_extension("wbin"));

            if let Err(e) = fs::write(&output_path, &result.binary) {
                eprintln!("error: failed to write {}: {}", output_path.display(), e);
                return Err("assembly failed".into());
            }

            if args.display.verbose {
                println!(
                    "Assembled {} bytes to {}",
                    result.binary.len(),
                    output_path.display()
                );
            }

            Ok(())
        }
        Err(e) => {
            let emitter = DiagnosticEmitter::new(&source_name, &source);
            emitter.emit_error(&e, &mut std::io::stderr());
            Err("assembly failed".into())
        }
    }
}
