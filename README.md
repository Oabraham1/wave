# WAVE: Wide Architecture Virtual Encoding

**The ARM of GPU computing. One binary, any GPU.**

[![arXiv](https://img.shields.io/badge/arXiv-2603.28793-b31b1b)](https://arxiv.org/abs/2603.28793)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19163452-blue)](https://doi.org/10.5281/zenodo.19163452)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/wave-gpu)](https://pypi.org/project/wave-gpu/)

WAVE is a vendor-neutral GPU instruction set architecture. Write GPU code once, run it on NVIDIA, AMD, Apple, and Intel GPUs unchanged. The same binary produces identical results on all four vendors. ARM defines what a CPU is so multiple vendors can build compatible chips. WAVE does the same for GPU computation.

## Highlights

- **11** hardware-invariant primitives across 4 GPU vendors
- **34,000** lines of Rust across 10 crates
- **618+** unit tests, **102/102** conformance tests passing
- **Verified** on Apple M4 Pro, NVIDIA T4, and AMD MI300X
- **3,587 GFLOPS** F32 matrix multiply on M4 Pro (53.5% of Apple MPS)
- **89.29%** CIFAR-10 accuracy via PyTorch integration, matching native exactly

## Install

```bash
pip install wave-gpu
```

Or build from source:

```bash
git clone https://github.com/Oabraham1/wave.git
cd wave
for crate in wave-decode wave-asm wave-dis wave-emu wave-compiler wave-metal wave-ptx wave-hip wave-sycl wave-runtime; do
  (cd $crate && cargo build --release)
done
```

## Quickstart

```python
import wave_gpu as wg

device = wg.device()
print(f"Running on: {device}")

a = wg.array([1.0, 2.0, 3.0, 4.0])
b = wg.array([5.0, 6.0, 7.0, 8.0])
out = wg.zeros(4)

print(f"a: {a}")
print(f"b: {b}")
```

## Architecture

```
Source Code (Python / Rust / C++ / TypeScript)
  |
  v
wave-compiler ──> WAVE Binary (.wbin) ──> wave-emu (reference emulator)
                        |
           ┌────────────┼────────────┐
           v            v            v
       wave-metal   wave-ptx    wave-hip    wave-sycl
       (Apple MSL)  (NVIDIA)    (AMD ROCm)  (Intel oneAPI)
           |            |            |            |
           v            v            v            v
        Apple GPU    NVIDIA GPU   AMD GPU    Intel GPU
```

| Crate | Purpose |
|-------|---------|
| `wave-decode` | Shared instruction decoder and binary format |
| `wave-asm` | Assembler (.wave text to .wbin binary) |
| `wave-dis` | Disassembler (.wbin binary to .wave text) |
| `wave-emu` | Reference emulator |
| `wave-compiler` | Multi-language compiler (Python/Rust/C++/TS to .wbin) |
| `wave-metal` | Apple Metal backend |
| `wave-ptx` | NVIDIA PTX backend |
| `wave-hip` | AMD HIP backend |
| `wave-sycl` | Intel SYCL backend |
| `wave-runtime` | SDK runtime with in-process compilation and kernel cache |
| `sdk/python` | Python SDK (`pip install wave-gpu`) |

Each crate builds independently. No Cargo workspace.

## Benchmarks

Auto-tuned results on Apple M4 Pro at 4096x4096 matrix size (MPS baseline: 6,710 GFLOPS):

| Kernel | F32 GFLOPS | F16 GFLOPS | % of MPS |
|--------|-----------|-----------|----------|
| Blocked GEMM | 3,587 | 4,049 | 53.5% |
| Fused GEMM+bias+ReLU | 3,562 | 4,027 | 53.1% |
| Fused GEMM+bias+GELU | 3,514 | -- | 52.4% |

Cross-vendor hardware verification:

| Vendor | GPU | Status |
|--------|-----|--------|
| Apple | M4 Pro | Verified |
| NVIDIA | T4 | Verified |
| AMD | MI300X | Verified |
| Intel | Arc | Pending |

## Papers

- [Toward a Universal GPU Instruction Set Architecture: A Cross-Vendor Analysis of Hardware-Invariant Computational Primitives in Parallel Processors](https://doi.org/10.5281/zenodo.19163452) (Zenodo, 2026)
- arXiv preprint: [2603.28793](https://arxiv.org/abs/2603.28793)
- Under review: International Journal of Parallel Programming (IJPP), April 2026
- Venue targets: ASPLOS 2027, CGO 2026, MLSys, CAV

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the fork-based workflow, code standards, and testing requirements.

## Authors

- **Ojima Abraham** (Franklin & Marshall College) [![ORCID](https://img.shields.io/badge/ORCID-0009--0005--8667--4771-green)](https://orcid.org/0009-0005-8667-4771)
- **Onyinye Okoli** (Cornell University) [![ORCID](https://img.shields.io/badge/ORCID-0009--0001--3374--2890-green)](https://orcid.org/0009-0001-3374-2890)

## License

Apache License, Version 2.0. See [LICENSE](LICENSE) for terms.

## Acknowledgments

Asahi Linux reverse engineering team, Dougall Johnson (GPU microarchitecture documentation), AMD GPUOpen, Google Colab (NVIDIA T4 verification), DigitalOcean (AMD MI300X verification).
