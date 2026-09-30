#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build and measure the fixed WAVE GEMM suite and the MPS reference."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from run_autotune import REPO_ROOT, SCRIPT_DIR, nonnegative, positive, run_logged, sha256, tool_output

KERNELS = ("gemm_blocked", "gemm_blocked_f16", "gemm_bias_relu", "gemm_bias_gelu", "gemm_bias_relu_f16")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="128,256,512,1024,2048,4096")
    parser.add_argument("--warmup", type=nonnegative, default=5)
    parser.add_argument("--iterations", type=positive, default=20)
    parser.add_argument("--seed", type=nonnegative, default=42)
    parser.add_argument("--work-dir", type=Path, help="new artifact directory")
    parser.add_argument("--prepare-only", action="store_true", help="build artifacts without GPU execution")
    parser.add_argument("--run-prepared", type=Path, help="run the exact command in a prepared manifest")
    args = parser.parse_args()
    if args.run_prepared:
        work = args.run_prepared.resolve()
        manifest = json.loads((work / "manifest.json").read_text())
        result_path = Path(manifest["command"][manifest["command"].index("--output") + 1])
        if result_path.exists():
            parser.error(f"results already exist: {result_path}")
        for path, digest in manifest["artifact_sha256"].items():
            if sha256(work / path) != digest:
                parser.error(f"prepared artifact changed: {path}")
        command = manifest["command"]
    else:
        try:
            sizes = [positive(value) for value in args.sizes.split(",")]
            if len(set(sizes)) != len(sizes) or any(size % 128 or size > 16384 for size in sizes):
                parser.error("sizes must be distinct multiples of 128, no larger than 16384")
        except (ValueError, argparse.ArgumentTypeError) as error:
            parser.error(str(error))
        if args.seed > 2**64 - 1:
            parser.error("seed must fit UInt64")
        if args.work_dir:
            work = args.work_dir.resolve()
            work.mkdir(parents=True, exist_ok=False)
        else:
            work = Path(tempfile.mkdtemp(prefix="wave-metal-bench-"))
        print(f"Artifacts: {work}", file=sys.stderr)
        target = work / "cargo-target"
        for package in ("wave-asm", "wave-metal"):
            run_logged(["cargo", "build", "--offline", "--locked", "--release", "--manifest-path",
                        REPO_ROOT / package / "Cargo.toml", "--target-dir", target], work / f"build-{package}.log")
        sources = work / "sources"
        sources.mkdir()
        for name in KERNELS:
            assembly = sources / f"{name}.wave"
            shutil.copy2(REPO_ROOT / "tests/ci/kernels" / f"{name}.wave", assembly)
            run_logged([target / "release/wave-asm", assembly, "-o", sources / f"{name}.wbin"],
                       work / f"{name}.assemble.log")
            run_logged([target / "release/wave-metal", sources / f"{name}.wbin", "-o", sources / f"{name}.metal"],
                       work / f"{name}.codegen.log")
        for source in (SCRIPT_DIR / "BenchmarkSupport.swift", REPO_ROOT / "benchmarks/metal_benchmark.swift"):
            shutil.copy2(source, sources / source.name)
        binary = work / "metal_benchmark"
        run_logged(["swiftc", "-O", "-Xcc", "-DACCELERATE_NEW_LAPACK", sources / "BenchmarkSupport.swift",
                    sources / "metal_benchmark.swift", "-framework", "Metal", "-framework", "MetalPerformanceShaders",
                    "-framework", "Accelerate", "-o", binary], work / "build-harness.log")
        command = [str(binary), "--metal-dir", str(sources), "--sizes", args.sizes,
                   "--warmup", str(args.warmup), "--iterations", str(args.iterations), "--seed", str(args.seed),
                   "--output", str(work / "results.jsonl")]
        manifest = {
            "schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_revision": tool_output(["git", "rev-parse", "HEAD"]),
            "git_status": tool_output(["git", "status", "--porcelain"]),
            "platform": platform.platform(), "swift": tool_output(["swiftc", "--version"]),
            "rustc": tool_output(["rustc", "--version"]), "command": command,
            "case_order": [*KERNELS, "mps_f32"], "size_order": sizes,
            "artifact_sha256": {
                str(path.relative_to(work)): sha256(path) for path in [*sources.iterdir(), binary]
            },
        }
        (work / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        if args.prepare_only:
            print(f"Prepared: {work}", file=sys.stderr)
            return 0
    with (work / "run.log").open("w") as diagnostics:
        result = subprocess.run(command, cwd=REPO_ROOT, stdout=diagnostics, stderr=subprocess.STDOUT)
    print(f"Results: {work / 'results.jsonl'}", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
