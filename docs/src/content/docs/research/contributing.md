---
title: Contributing
description: Build, test, and contribute to WAVE.
---

Contributions are managed in the [WAVE repository](https://github.com/Oabraham1/wave).

## Build and test

Follow [Installation](/getting-started/installation/) to build the toolchain. Each package has its own manifest. From the repository root, run the relevant package's checks:

```sh
cargo fmt --manifest-path wave-compiler/Cargo.toml --check
cargo clippy --manifest-path wave-compiler/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path wave-compiler/Cargo.toml
```

For changes to instruction encoding, execution, or translation, also run the integration checks:

```sh
bash tests/ci/run_emulator_tests.sh
bash tests/ci/run_backend_tests.sh
cargo run --manifest-path tests/spec-verification/Cargo.toml
```

CI checks formatting, linting, package tests, SDK tests, and emulator and backend integration tests. The [workflow](https://github.com/Oabraham1/wave/blob/main/.github/workflows/ci.yml) contains the commands for each language.

## Changes

Compiler and backend changes must preserve the ISA's behavior. Include a test that exercises the affected behavior. Performance claims need measurements and correctness checks on the named hardware. See [Hardware verification](/research/hardware-verification/) for the benchmark methods.

Specification proposals should identify the operation or rule being changed and explain its effect on each backend. SDK and documentation changes should include examples that work with the current interfaces.

## Pull requests

1. Create a branch from `main`.
2. Make the change and run the relevant checks.
3. Describe the problem, resulting behavior, and validation in the pull request.
4. Address CI failures before requesting review.

Use `rustfmt` defaults, document public interfaces, and fix Clippy warnings at their source.
