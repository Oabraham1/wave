# wave-compiler

Multi-language GPU kernel compiler targeting the WAVE ISA.

Compiles supported Python, Rust, C++, and TypeScript kernel syntax into WAVE `.wbin` files. Uses SSA for optimization and interference graph coloring for register allocation.

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

1. **Frontend** - Language-specific parser produces HIR, High-Level IR
2. **HIR → MIR** - Lowering to SSA form with phi nodes and CFG
3. **Optimization** - DCE, CSE, SCCP, LICM, strength reduction, mem2reg, loop unrolling, CFG simplification
4. **MIR → LIR** - Instruction selection with structured control flow emission
5. **Register Allocation + Emission** - Bounded graph coloring and WAVE binary encoding. Allocation errors if live values cannot fit; the active path does not insert spills.

## License

Apache 2.0 - see [LICENSE](../LICENSE)
