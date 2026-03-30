# wave-compiler

Multi-language GPU kernel compiler targeting the WAVE ISA.

Compiles GPU kernels written in Python, Rust, C++, or TypeScript into WAVE binary files (.wbin). Includes an SSA-based intermediate representation, optimization passes, and a Chaitin-Briggs register allocator.

## Build

```bash
cargo build --release
```

## Usage

```bash
wave-compiler input.py -o output.wbin --lang python
wave-compiler input.rs -o output.wbin --lang rust
wave-compiler input.cpp -o output.wbin --lang cpp
wave-compiler input.ts -o output.wbin --lang typescript
```

## Architecture

Source code passes through five stages:

1. **Frontend** - Language-specific parser produces HIR (High-Level IR)
2. **HIR → MIR** - Lowering to SSA form with phi nodes and CFG
3. **Optimization** - DCE, CSE, SCCP, LICM, strength reduction, mem2reg, loop unrolling, CFG simplification
4. **MIR → LIR** - Instruction selection with structured control flow emission
5. **Register Allocation + Emission** - Chaitin-Briggs graph coloring, WAVE binary encoding

## License

Apache 2.0 - see [LICENSE](../LICENSE)
