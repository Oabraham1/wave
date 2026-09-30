---
title: "Research papers"
description: "The WAVE preprint, implementation, and GPU experiments."
---

## Preprint

Toward a Universal GPU Instruction Set Architecture: A Cross-Vendor Analysis of Hardware-Invariant Computational Primitives in Parallel Processors

Ojima Abraham and Onyinye Okoli. The preprint derives common GPU execution primitives and presents the WAVE architecture. Its experiments use hand-written kernels to examine these primitives across vendors.

- [arXiv v1](https://arxiv.org/abs/2603.28793v1)
- [Zenodo record](https://doi.org/10.5281/zenodo.19163452)

## GPU experiments

The [GPU benchmarks](https://github.com/Oabraham1/wave/tree/main/benchmarks/gpu) compare WAVE binaries with matched native kernels on Apple and NVIDIA hardware. They check complete outputs against independent references and measure execution with GPU timestamps. See [Hardware verification](/research/hardware-verification/) for the workloads, methods, and results.

## Implementation

The [repository](https://github.com/Oabraham1/wave) contains the source compiler, assembler, emulator, backend translators, runtime, and SDKs. The source compiler emits WAVE binaries. The backends emit Metal Shading Language, PTX, HIP source, or SYCL source for vendor toolchains. The emulator checks functional execution; it does not predict hardware cycle counts.

## Citation

```bibtex
@misc{abraham2026wave,
  title = {Toward a Universal GPU Instruction Set Architecture:
           A Cross-Vendor Analysis of Hardware-Invariant
           Computational Primitives in Parallel Processors},
  author = {Ojima Abraham and Onyinye Okoli},
  year = {2026},
  eprint = {2603.28793},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2603.28793}
}
```
