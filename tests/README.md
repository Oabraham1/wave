# WAVE Tests

## spec-verification/

Specification conformance tests verifying every normative claim in the WAVE spec. Runs kernels on the emulator and checks correctness.

```bash
cd spec-verification && cargo run
```

## ci/

CI integration test scripts run by GitHub Actions. Tests the full pipeline: assemble/compile → emulate → verify output.

```bash
./ci/run_emulator_tests.sh
./ci/run_backend_tests.sh
```

## License

Apache 2.0 - see [LICENSE](../LICENSE)
