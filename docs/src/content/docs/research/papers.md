---
title: Research Papers
description: Publications, preprints, and source code for the WAVE research project.
---

WAVE is an active research project. This page collects links to the published paper, pending publications, and the open-source implementation.

## Primary Publication

**Toward a Universal GPU Instruction Set Architecture: A Cross-Vendor Analysis of Hardware-Invariant Computational Primitives in Parallel Processors**

The paper presents the systematic analysis of over 5,000 pages of vendor ISA documentation across 16 microarchitectures from Apple, NVIDIA, AMD, and Intel. It identifies 11 categories of hardware-invariant primitives that every major GPU architecture provides, and introduces WAVE as a portable binary encoding built on top of those primitives.

The research demonstrates end-to-end correctness by compiling a single WAVE binary and executing it on three distinct GPUs - Apple M4 Pro (Metal), NVIDIA T4 (Turing/PTX), and AMD MI300X (CDNA 3/GCN ISA) - producing identical numerically verified output on all targets.

### Zenodo (Open Access)

The paper is available on Zenodo under DOI [10.5281/zenodo.19163452](https://doi.org/10.5281/zenodo.19163452).

### ASPLOS 2027

The paper is in preparation for submission to the 32nd ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS 2027).

## Source Code

The full WAVE implementation - compiler, emulator, runtime, and multi-language SDKs - is open source on GitHub:

- **Repository:** [github.com/Oabraham1/wave](https://github.com/Oabraham1/wave)
- **License:** See the repository for license details.

The repository contains:

- `wave-compiler` - the ahead-of-time compiler that translates WAVE binaries into vendor-native GPU instructions (Metal IR, PTX, GCN ISA).
- `wave-emu` - a cycle-approximate emulator for WAVE binaries.
- `wave-runtime` - the host-side runtime that manages device discovery, memory, and kernel dispatch.
- Language SDKs for Rust, Python, TypeScript, and C++.

## Citing WAVE

If you use WAVE in your research, please cite the Zenodo publication:

```bibtex
@software{wave2026,
  title     = {Toward a Universal GPU Instruction Set Architecture:
               A Cross-Vendor Analysis of Hardware-Invariant
               Computational Primitives in Parallel Processors},
  doi       = {10.5281/zenodo.19163452},
  publisher = {Zenodo},
  url       = {https://doi.org/10.5281/zenodo.19163452}
}
```
