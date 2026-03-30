# CI Integration Tests

Automated tests that verify the full WAVE pipeline by running compiled kernels on the emulator and comparing output against expected values derived from real GPU hardware (Apple M4 Pro, NVIDIA T4, AMD MI300X).

## Scripts

- `run_emulator_tests.sh` - 11 tests: assembly kernels + compiled kernels from all 4 frontends + loop kernel
- `run_backend_tests.sh` - 4 tests: Metal, PTX, HIP, SYCL codegen verification

## Kernels

Test kernel sources in `kernels/`:
- `scalar_multiply.wave` - c[i] = a[i] * 5.0
- `vector_sub.wave` - c[i] = a[i] - b[i]
- `bounds_check.wave` - c[i] = a[i] + b[i] for i < n, with predication

## License

Apache 2.0 - see [LICENSE](../../LICENSE)
