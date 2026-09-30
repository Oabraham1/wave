# WAVE: Wide Architecture Virtual Encoding

[![arXiv](https://img.shields.io/badge/arXiv-2603.28793-b31b1b)](https://arxiv.org/abs/2603.28793)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19163452-blue)](https://doi.org/10.5281/zenodo.19163452)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/wave-gpu)](https://pypi.org/project/wave-gpu/)

WAVE is a GPU instruction set with backends for NVIDIA, AMD, Apple, and Intel. Compile a kernel to a `.wbin` file, then translate it for the target GPU. Hardware experiments execute the same binaries on Apple M4 Pro and NVIDIA Tesla T4. AMD and Intel hardware execution is outside this evaluation.

## Highlights

- Explicit registers, structured control flow, and wave operations
- Shared binary format for the emulator and backend translators
- GPU experiments with independent references and matched native kernels

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
| `wave-asm` | Assembler; .wave text to .wbin binary |
| `wave-dis` | Disassembler; .wbin binary to .wave text |
| `wave-emu` | Reference emulator |
| `wave-compiler` | Multi-language compiler; Python/Rust/C++/TS to .wbin |
| `wave-metal` | Apple Metal backend |
| `wave-ptx` | NVIDIA PTX backend |
| `wave-hip` | AMD HIP backend |
| `wave-sycl` | Intel SYCL backend |
| `wave-runtime` | SDK runtime with in-process compilation and kernel cache |
| `sdk/python` | Python SDK; `pip install wave-gpu` |

Each crate builds independently. No Cargo workspace.

## Benchmarks

The [GPU benchmarks](benchmarks/gpu/) compare WAVE binaries with matched native Metal and CUDA kernels. Run the Metal benchmarks on macOS 15 or later with an Apple GPU, Cargo, and the Swift command-line tools:

```sh
python3 benchmarks/gpu/run.py --output /private/tmp/wave-gpu-run
```

For a comparison with Slang 2026.19:

```sh
python3 benchmarks/gpu/run_comparison.py --output /private/tmp/wave-gpu-comparison \
  --slangc /path/to/slang-2026.19/bin/slangc
```

Package a Metal run for NVIDIA:

```sh
python3 benchmarks/gpu/nvidia/prepare.py --session /private/tmp/wave-gpu-run \
  --output /private/tmp/wave-nvidia
```

Upload the generated `nvidia_colab.ipynb` to Colab, select a GPU runtime, and run its cells. Use a new output directory for each run. Results include complete output checks and raw timing samples.

The [GEMM suite](benchmarks/autotune/README.md) compares explicit WAVE kernels with Metal Performance Shaders. Full output validation uses Accelerate SGEMM. Each run writes its configuration and timing samples to the selected output directory.

## Papers

- [Toward a Universal GPU Instruction Set Architecture: A Cross-Vendor Analysis of Hardware-Invariant Computational Primitives in Parallel Processors](https://doi.org/10.5281/zenodo.19163452), Zenodo, 2026
- arXiv preprint: [2603.28793](https://arxiv.org/abs/2603.28793)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the fork-based workflow, code standards, and testing requirements.

## Authors

- **Ojima Abraham**, Franklin & Marshall College, [![ORCID](https://img.shields.io/badge/ORCID-0009--0005--8667--4771-green)](https://orcid.org/0009-0005-8667-4771)
- **Onyinye Okoli**, Cornell University, [![ORCID](https://img.shields.io/badge/ORCID-0009--0001--3374--2890-green)](https://orcid.org/0009-0001-3374-2890)

## License

Apache License, Version 2.0. See [LICENSE](LICENSE) for terms.

## Acknowledgments

Asahi Linux reverse engineering team, Dougall Johnson, GPU microarchitecture documentation, AMD GPUOpen, Google Colab, NVIDIA T4 verification, DigitalOcean, AMD MI300X verification.
