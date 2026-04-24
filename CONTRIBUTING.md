# Contributing to WAVE

Thank you for your interest in contributing to WAVE. This document covers the workflow, code standards, and testing requirements for all contributions.

## Getting Started

1. Fork [https://github.com/Oabraham1/wave](https://github.com/Oabraham1/wave) on GitHub
2. Clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/wave.git
cd wave
```

3. Build all crates:

```bash
for crate in wave-decode wave-asm wave-dis wave-emu wave-compiler wave-metal wave-ptx wave-hip wave-sycl wave-runtime; do
  echo "Building $crate..."
  (cd $crate && cargo build --release)
done
```

4. Run all tests:

```bash
for crate in wave-decode wave-asm wave-dis wave-emu wave-compiler wave-metal wave-ptx wave-hip wave-sycl wave-runtime; do
  echo "Testing $crate..."
  (cd $crate && cargo test)
done
```

5. Run CI integration tests:

```bash
bash tests/ci/run_emulator_tests.sh
bash tests/ci/run_backend_tests.sh
```

## Workflow

Fork and branch. Never commit directly to main.

1. Fork `https://github.com/Oabraham1/wave` to your GitHub account
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/wave.git`
3. Create a branch: `git checkout -b fix/description-of-change`
4. Make your changes
5. Run the full test suite (see [Testing](#testing) below)
6. Push to your fork: `git push origin fix/description-of-change`
7. Open a Pull Request against `Oabraham1/wave:main`

Branch naming conventions:

- `fix/thing-you-fixed`
- `feat/thing-you-added`
- `chore/thing-you-cleaned`
- `docs/thing-you-documented`

## Code Standards

These are enforced by CI. PRs that violate them will not be merged.

### Licensing

Every source file must begin with a license header.

Rust files:

```rust
// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0
```

Python files:

```python
# Copyright 2026 Ojima Abraham
# SPDX-License-Identifier: Apache-2.0
```

WAVE assembly files:

```asm
; SPDX-License-Identifier: Apache-2.0
```

### File-Level Documentation

Every source file must have a comment block immediately after the SPDX header explaining what the file does, how it works, and why it exists.

For Rust, use `//!` doc comments:

```rust
// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! GPU detection and device enumeration for the WAVE runtime.
//!
//! Probes the system for available GPU hardware. Returns all detected
//! GPUs with detailed capability information for multi-GPU workloads.
```

For Python, use a module-level docstring:

```python
# Copyright 2026 Ojima Abraham
# SPDX-License-Identifier: Apache-2.0

"""GPU detection for the WAVE Python SDK."""
```

### No Inline Comments

Do not put comments inside function bodies. Code must be self-documenting through clear naming. If you need a comment to explain what code does, rename the variables and functions until you don't.

### Formatting

- Rust: `cargo fmt` (standard rustfmt)
- Python: `ruff format` and `ruff check`
- WAVE assembly: 4-space indentation for instructions inside `.kernel`/`.end` blocks

### Clippy

All Rust code must pass `cargo clippy -- -D warnings` with zero warnings. Do not add `#[allow(...)]` suppressions unless absolutely necessary. If you must add one, explain why in the PR description.

### No Dead Code

Do not commit unused functions, structs, constants, or imports. Do not prefix things with `_` to suppress unused warnings unless the parameter is genuinely unused by design (e.g., a trait method you must implement but don't need the argument).

## Architecture

```
wave-decode     Shared instruction decoder (foundation crate)
wave-asm        Assembler (.wave -> .wbin)
wave-dis        Disassembler (.wbin -> .wave)
wave-emu        Reference emulator
wave-compiler   Multi-language compiler (Python/Rust/C++/TS -> .wbin)
wave-metal      Apple Metal backend
wave-ptx        NVIDIA PTX backend
wave-hip        AMD HIP backend
wave-sycl       Intel SYCL backend
wave-runtime    SDK runtime (ties everything together)
```

There is no Cargo workspace. Each crate builds independently. `wave-decode` is the only shared dependency.

## Testing

Before opening a PR, run all of these. All must pass.

### Unit Tests (all 10 crates)

```bash
for crate in wave-decode wave-asm wave-dis wave-emu wave-compiler wave-metal wave-ptx wave-hip wave-sycl wave-runtime; do
  echo "=== $crate ==="
  (cd $crate && cargo test)
done
```

### Clippy (all 10 crates)

```bash
for crate in wave-decode wave-asm wave-dis wave-emu wave-compiler wave-metal wave-ptx wave-hip wave-sycl wave-runtime; do
  echo "=== $crate ==="
  (cd $crate && cargo clippy -- -D warnings)
done
```

### CI Integration Tests

```bash
bash tests/ci/run_emulator_tests.sh
bash tests/ci/run_backend_tests.sh
```

### Spec Verification

If you changed the encoding, decoder, assembler, or emulator:

```bash
cd tests/spec-verification && cargo run
```

If a test fails, fix your code, not the test.

## What to Contribute

### Good First Issues

- Add a new WAVE assembly kernel to `tests/ci/kernels/`
- Improve error messages in wave-asm diagnostics
- Add a missing instruction to a backend (check TODOs in the code)
- Write documentation for the docs site

### Larger Contributions

- New compiler optimization passes (add to `wave-compiler/src/optimize/`)
- New language frontend (add to `wave-compiler/src/frontend/`)
- Performance improvements to existing kernels
- Intel GPU verification (requires Arc A380 or equivalent)

### Please Discuss First

Open an issue before working on:

- ISA changes (open an issue with the proposal)
- New instruction categories
- Changes to the binary encoding format
- Anything that affects spec verification tests

## Commit Messages

Format: `type: short description`

Types: `fix`, `feat`, `chore`, `docs`, `perf`, `test`, `refactor`

Examples:

- `fix: restore predicate encoding in word0`
- `feat: add auto-tuning framework`
- `chore: remove dead code across all crates`
- `docs: update spec to v0.4`
- `perf: replace subprocess compilation with in-process library calls`

## Pull Request Checklist

Before submitting, verify:

- [ ] All 10 crates pass `cargo test`
- [ ] All 10 crates pass `cargo clippy -- -D warnings`
- [ ] All 10 crates pass `cargo fmt --check`
- [ ] CI emulator tests pass (22/22)
- [ ] CI backend tests pass (4/4)
- [ ] Spec verification passes (102/102) if encoding/decoder/asm/emu changed
- [ ] SPDX headers on all new files
- [ ] File-level documentation on all new files
- [ ] No inline comments in function bodies
- [ ] No dead code or unused suppressions
- [ ] Commit messages follow the `type: description` format
