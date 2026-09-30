# Contributing to WAVE

Fork [the repository](https://github.com/Oabraham1/wave), create a branch, and open a pull request against `Oabraham1/wave:main`. Keep each change focused. Include the problem, the resulting behavior, and the checks you ran.

## Build and test

Build each package independently. There is no Cargo workspace.

```bash
for crate in wave-decode wave-asm wave-dis wave-emu wave-compiler wave-metal wave-ptx wave-hip wave-sycl wave-runtime sdk/rust tests/spec-verification; do
  cargo fmt --manifest-path "$crate/Cargo.toml" --check
  cargo clippy --manifest-path "$crate/Cargo.toml" --all-targets -- -D warnings
  cargo test --manifest-path "$crate/Cargo.toml"
done
```

Run the integration checks and specification suite:

```bash
bash tests/ci/run_emulator_tests.sh
bash tests/ci/run_backend_tests.sh
cargo run --manifest-path tests/spec-verification/Cargo.toml
```

Build the documentation after changing a page:

```bash
cd docs
npm ci
npm run build
```

## Code standards

Follow [AGENTS.md](AGENTS.md). Use `rustfmt`, propagate errors with `Result`, and document public APIs. Fix Clippy findings in the implementation. Do not add lint suppressions.

Comments should explain an invariant or a decision that the code cannot express. Every unsafe block needs a `SAFETY` comment. Keep prose short and direct. Avoid em dashes and parenthetical asides. Preserve punctuation required by code and formulas.

Start source files with a license header:

```rust
// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0
```

Use `#` for Python and shell headers, and `;` for WAVE assembly headers. Add a short module description after the header. Format Python with `ruff format` and check it with `ruff check`.

## Packages

| Package | Purpose |
| --- | --- |
| `wave-decode` | Shared instruction and WBIN decoder |
| `wave-asm` | WAVE assembler |
| `wave-dis` | WAVE disassembler |
| `wave-emu` | Reference emulator |
| `wave-compiler` | Python, Rust, C++, and TypeScript compiler |
| `wave-metal` | Apple Metal backend |
| `wave-ptx` | NVIDIA PTX backend |
| `wave-hip` | AMD HIP backend |
| `wave-sycl` | Intel SYCL backend |
| `wave-runtime` | Shared SDK runtime |
| `sdk/rust` | Rust SDK |
| `tests/spec-verification` | ISA conformance checks |

## Scope

Bug fixes, backend support, error messages, examples, tests, and documentation are useful contributions. Measure performance changes and verify their output against the ISA.

Open an issue before changing the ISA, instruction categories, binary encoding, or conformance requirements. Keep implementation changes consistent with the specification.

Use commit messages in the form `type: short description`, with `fix`, `feat`, `chore`, `docs`, `perf`, `test`, or `refactor`.
