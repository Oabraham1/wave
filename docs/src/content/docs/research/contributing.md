---
title: Contributing
description: How to build WAVE from source, run tests, and contribute to the project.
---

WAVE is open source and hosted on GitHub at [github.com/Oabraham1/wave](https://github.com/Oabraham1/wave). Contributions are welcome across the entire project.

## Building from Source

WAVE is written in Rust. You need a working Rust toolchain installed via [rustup](https://rustup.rs/).

```bash
git clone https://github.com/Oabraham1/wave.git
cd wave
cargo build --release
```

This builds all crates in the workspace: the compiler (`wave-compiler`), the emulator (`wave-emu`), the runtime (`wave-runtime`), and the language SDKs.

## Running Tests

Each crate has its own test suite. Run tests for a specific crate from the workspace root:

```bash
cargo test -p wave-compiler
cargo test -p wave-emu
cargo test -p wave-runtime
```

Or run the full test suite:

```bash
cargo test
```

## Continuous Integration

The CI pipeline runs on every pull request and checks:

- **Lint** - `cargo clippy` across all crates.
- **Format** - `cargo fmt --check` to enforce consistent style.
- **SDK tests** - test suites for all language SDKs (Rust, Python, TypeScript, C++).

All CI checks must pass before a pull request can be merged.

## Areas of Contribution

There are several areas where contributions have high impact:

### New Backends

WAVE currently compiles to Metal IR, PTX, and GCN ISA. Adding support for additional targets - particularly Intel Xe - would expand hardware coverage. Backend work lives in `wave-compiler`.

### Compiler Optimizations

The compiler performs ahead-of-time translation from WAVE's portable encoding to vendor-native instructions. There are opportunities for optimization passes such as instruction scheduling, register allocation improvements, and vendor-specific peephole optimizations.

### Hardware Verification

If you have access to GPU hardware that WAVE targets - especially Intel Xe - running the verification suite and reporting results is a valuable contribution. See the [Hardware Verification](/research/hardware-verification/) page for the methodology.

### Spec Improvements

The WAVE specification defines the portable instruction set and binary format. Proposals to extend the spec with new primitives, refine edge-case semantics, or improve clarity are welcome as issues or pull requests.

### Documentation

Improvements to these docs, additional examples, and tutorials for the language SDKs are always appreciated.

## Code Style

- Rust code follows `rustfmt` defaults. Run `cargo fmt` before committing.
- All public items should have doc comments.
- Address `clippy` warnings - CI will reject code that produces them.

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes, ensuring that `cargo test`, `cargo fmt --check`, and `cargo clippy` all pass locally.
3. Write a clear commit message describing what the change does and why.
4. Open a pull request against `main`. The PR description should summarize the motivation and any design decisions.
5. CI will run automatically. Address any failures before requesting review.
