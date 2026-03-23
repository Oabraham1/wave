// Copyright (c) 2026 Ojima Abraham. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

// CLI entry point for wave-dis. Parses command-line arguments, reads WBIN
// files, invokes the disassembler, and writes output to stdout or file.

use clap::Parser;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use wave_dis::{disassemble_wbin, DisassemblyOptions};

#[derive(Parser)]
#[command(name = "wave-dis")]
#[command(about = "WAVE disassembler - converts .wbin files to assembly text")]
#[command(version)]
struct Args {
    #[arg(help = "Input .wbin file")]
    input: PathBuf,

    #[arg(short, long, help = "Output file (default: stdout)")]
    output: Option<PathBuf>,

    #[arg(long, help = "Show byte offsets for each instruction")]
    offsets: bool,

    #[arg(long, help = "Show raw hex encoding alongside assembly")]
    raw: bool,

    #[arg(long = "no-directives", help = "Omit .kernel/.registers/.end directives")]
    no_directives: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let data = match fs::read(&args.input) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: failed to read '{}': {}", args.input.display(), e);
            return ExitCode::from(1);
        }
    };

    let options = DisassemblyOptions {
        show_offsets: args.offsets,
        show_raw: args.raw,
        emit_directives: !args.no_directives,
    };

    let lines = match disassemble_wbin(&data, &options) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };

    let output_text = lines.join("\n");

    if let Some(output_path) = args.output {
        match fs::write(&output_path, format!("{output_text}\n")) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("error: failed to write '{}': {}", output_path.display(), e);
                return ExitCode::from(1);
            }
        }
    } else {
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        if let Err(e) = writeln!(handle, "{output_text}") {
            eprintln!("error: failed to write to stdout: {e}");
            return ExitCode::from(1);
        }
    }

    ExitCode::SUCCESS
}
