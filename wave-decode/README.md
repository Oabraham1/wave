# wave-decode

Shared instruction decoder for the WAVE ISA. This crate provides:

- **Opcode definitions** matching `wave-asm`'s encoding
- **Instruction decoding** from binary to structured Rust types
- **WBIN container parsing** for reading `.wbin` files

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
wave-decode = { path = "../wave-decode" }
```

### Decoding Instructions

```rust
use wave_decode::{Decoder, decode_all};

// Decode from raw bytes
let code: &[u8] = &[0x01, 0x01, 0x00, 0xFC]; // halt instruction
let mut decoder = Decoder::new(code);

while decoder.has_more() {
    let instruction = decoder.decode_next()?;
    println!("{}: {}", instruction.offset, instruction.mnemonic());
}

// Or decode all at once
let instructions = decode_all(code)?;
```

### Reading WBIN Files

```rust
use wave_decode::WbinFile;

let data = std::fs::read("kernel.wbin")?;
let wbin = WbinFile::parse(&data)?;

println!("Kernels: {}", wbin.kernel_count());
for kernel in &wbin.kernels {
    println!("  {} - {} registers, workgroup [{}, {}, {}]",
        kernel.name,
        kernel.register_count,
        kernel.workgroup_size[0],
        kernel.workgroup_size[1],
        kernel.workgroup_size[2],
    );
}

// Decode a specific kernel
if let Some(code) = wbin.kernel_code(0) {
    let instructions = wave_decode::decode_all(code)?;
    // ...
}
```

## Architecture

The crate is organized into four modules:

- `opcodes` - Constants and enums for all WAVE opcodes and modifiers
- `instruction` - `DecodedInstruction` and `Operation` types
- `decoder` - The `Decoder` struct that reads binary instructions
- `wbin` - WBIN container format parser

## WBIN Format

The WBIN format is a simple container:

| Offset | Size | Description |
|--------|------|-------------|
| 0x00   | 4    | Magic "WAVE" |
| 0x04   | 2    | Version (0x0001) |
| 0x06   | 2    | Flags (reserved) |
| 0x08   | 4    | Code section offset |
| 0x0C   | 4    | Code section size |
| 0x10   | 4    | Symbol table offset |
| 0x14   | 4    | Symbol table size |
| 0x18   | 4    | Metadata offset |
| 0x1C   | 4    | Metadata size |

Followed by the code section, symbol table (null-terminated strings), and kernel metadata.

## License

Apache-2.0
