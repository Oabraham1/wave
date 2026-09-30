#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate, compile, validate, and measure Metal GEMM configurations."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

from generate_kernel import KERNEL_NAMES, enumerate_configs, validate_config

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
CONFIG_FIELDS = ("tile_m", "tile_n", "tile_k", "block_m", "block_n", "prefetch")
SANITY_CONFIGS = [
    (64, 64, 4, 4, 4, False),
    (128, 128, 8, 8, 8, True),
    (128, 128, 4, 8, 8, True),
    (256, 256, 4, 8, 8, True),
    (64, 128, 4, 4, 8, True),
]


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def nonnegative(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def valid_result(record):
    """Accept timings only when full measured-size validation succeeded."""
    validation = record.get("validation", {})
    samples = record.get("gpu_seconds", [])
    size = record.get("bench_size", 0)
    return (
        record.get("status") == "ok"
        and record.get("correct") is True
        and validation.get("correct") is True
        and validation.get("coverage") == "full_output_at_measured_size"
        and validation.get("elements_checked") == size * size
        and validation.get("elements_expected") == size * size
        and validation.get("failed_elements") == 0
        and validation.get("nonfinite_elements") == 0
        and isinstance(record.get("gflops"), (int, float))
        and math.isfinite(record["gflops"])
        and record["gflops"] > 0
        and len(samples) == record.get("measurement", {}).get("timed_iterations")
        and bool(samples)
        and all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in samples)
    )


def config_tag(config):
    suffix = "p" if config["prefetch"] else "np"
    return (f"t{config['tile_m']}x{config['tile_n']}_k{config['tile_k']}"
            f"_b{config['block_m']}x{config['block_n']}_{suffix}")


def check_config(config, size):
    if set(config) != set(CONFIG_FIELDS):
        raise ValueError(f"config must contain exactly {', '.join(CONFIG_FIELDS)}")
    for field in CONFIG_FIELDS[:-1]:
        value = config[field]
        if type(value) is not int or value <= 0 or value & (value - 1):
            raise ValueError(f"{field} must be a positive power of two")
    if type(config["prefetch"]) is not bool:
        raise ValueError("prefetch must be a JSON boolean")
    for field in ("tile_m", "tile_n", "tile_k"):
        if size % config[field]:
            raise ValueError(f"benchmark size must be divisible by {field}")
    if config["tile_m"] % config["block_m"] or config["tile_n"] % config["block_n"]:
        raise ValueError("tile dimensions must be divisible by register block dimensions")
    valid, reason = validate_config(*(config[field] for field in CONFIG_FIELDS))
    if not valid:
        raise ValueError(reason)


def run_logged(command, path):
    """Keep diagnostics from every subprocess and propagate failures."""
    with path.open("w") as log:
        result = subprocess.run([str(arg) for arg in command], cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"command exited {result.returncode}: {command[0]}; diagnostics: {path}")


def tool_output(command):
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sanity", action="store_true", help="measure representative configurations")
    mode.add_argument("--full", action="store_true", help="measure all generated configurations")
    mode.add_argument("--config", type=json.loads, help="measure one configuration supplied as JSON")
    parser.add_argument("--variant", choices=KERNEL_NAMES, default="f32")
    parser.add_argument("--bench-size", type=positive, default=4096)
    parser.add_argument("--warmup", type=nonnegative, default=5)
    parser.add_argument("--iterations", type=positive, default=20)
    parser.add_argument("--seed", type=nonnegative, default=42)
    parser.add_argument("--order-seed", type=nonnegative, default=42)
    parser.add_argument("--work-dir", type=Path, help="new output directory; defaults to a unique temporary directory")
    args = parser.parse_args()
    if args.seed > 2**64 - 1:
        parser.error("seed must fit UInt64")
    if args.config is not None:
        configs = [args.config]
    elif args.full:
        configs = enumerate_configs()
    else:
        configs = [dict(zip(CONFIG_FIELDS, values)) for values in SANITY_CONFIGS]
    try:
        for config in configs:
            check_config(config, args.bench_size)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    random.Random(args.order_seed).shuffle(configs)
    if args.work_dir:
        work = args.work_dir.resolve()
        work.mkdir(parents=True, exist_ok=False)
    else:
        work = Path(tempfile.mkdtemp(prefix="wave-autotune-"))
    print(f"Artifacts: {work}", file=sys.stderr)
    target = work / "cargo-target"
    for package in ("wave-asm", "wave-metal"):
        run_logged(["cargo", "build", "--offline", "--locked", "--release", "--manifest-path",
                    REPO_ROOT / package / "Cargo.toml", "--target-dir", target], work / f"build-{package}.log")
    binary = work / "tune_benchmark"
    run_logged(["swiftc", "-O", "-Xcc", "-DACCELERATE_NEW_LAPACK", SCRIPT_DIR / "BenchmarkSupport.swift", SCRIPT_DIR / "tune_benchmark.swift",
                "-framework", "Metal", "-framework", "MetalPerformanceShaders", "-framework", "Accelerate",
                "-o", binary], work / "build-harness.log")
    manifest = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": tool_output(["git", "rev-parse", "HEAD"]),
        "git_status": tool_output(["git", "status", "--porcelain"]),
        "platform": platform.platform(),
        "swift": tool_output(["swiftc", "--version"]),
        "rustc": tool_output(["rustc", "--version"]),
        "variant": args.variant,
        "bench_size": args.bench_size,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "seed": args.seed,
        "order_seed": args.order_seed,
        "configuration_order": configs,
        "harness_sha256": sha256(binary),
        "generator_sha256": sha256(SCRIPT_DIR / "generate_kernel.py"),
        "support_sha256": sha256(SCRIPT_DIR / "BenchmarkSupport.swift"),
        "entry_source_sha256": sha256(SCRIPT_DIR / "tune_benchmark.swift"),
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    results = []
    with (work / "results.jsonl").open("w") as output:
        for order, config in enumerate(configs):
            tag = config_tag(config)
            record = {"schema_version": 1, "tag": tag, "variant": args.variant, "order_index": order, **config}
            source = work / f"{tag}.wave"
            assembled = work / f"{tag}.wbin"
            metal = work / f"{tag}.metal"
            try:
                command = [sys.executable, SCRIPT_DIR / "generate_kernel.py", "generate"]
                for field in CONFIG_FIELDS[:-1]:
                    command.extend(["--" + field.replace("_", "-"), config[field]])
                if config["prefetch"]:
                    command.append("--prefetch")
                command.extend(["--variant", args.variant, "-o", source])
                run_logged(command, work / f"{tag}.generate.log")
                run_logged([target / "release/wave-asm", source, "-o", assembled], work / f"{tag}.assemble.log")
                run_logged([target / "release/wave-metal", assembled, "-o", metal], work / f"{tag}.codegen.log")
                command = [binary, metal, KERNEL_NAMES[args.variant], config["tile_m"], config["tile_n"],
                           config["tile_n"] // config["block_n"], config["tile_m"] // config["block_m"],
                           int(args.variant not in ("f32", "f16")), args.bench_size,
                           args.warmup, args.iterations, args.seed, config["tile_k"]]
                with (work / f"{tag}.stderr.log").open("w") as diagnostics:
                    completed = subprocess.run([str(arg) for arg in command], cwd=REPO_ROOT,
                                               text=True, stdout=subprocess.PIPE, stderr=diagnostics)
                (work / f"{tag}.stdout.json").write_text(completed.stdout)
                measured = json.loads(completed.stdout)
                if not isinstance(measured, dict):
                    raise ValueError("harness output is not a JSON object")
                record.update(measured)
                record["process_exit_code"] = completed.returncode
                record["wave_source_sha256"] = sha256(source)
                record["wbin_sha256"] = sha256(assembled)
                if completed.returncode or not valid_result(record):
                    record["correct"] = False
                    if record.get("status") == "ok":
                        record["status"] = "invalid_result"
            except (OSError, RuntimeError, ValueError) as error:
                record.update(status="error", correct=False, error=str(error))
            output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            output.flush()
            results.append(record)
            if valid_result(record):
                print(f"{tag}: {record['gflops']:.1f} GFLOPS", file=sys.stderr)
            else:
                print(f"{tag}: {record.get('status', 'error')}; {record.get('error', 'output validation failed')}", file=sys.stderr)
    eligible = sorted((record for record in results if valid_result(record)), key=lambda item: item["gflops"], reverse=True)
    summary = {"schema_version": 1, "variant": args.variant, "bench_size": args.bench_size,
               "selection_metric": "highest median-sample GFLOPS among validated configurations",
               "results_path": str(work / "results.jsonl"), "best": eligible[0] if eligible else None,
               "complete": all(valid_result(record) for record in results)}
    (work / "selection.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if eligible:
        print(f"Selected: {eligible[0]['tag']}, {eligible[0]['gflops']:.1f} GFLOPS", file=sys.stderr)
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
