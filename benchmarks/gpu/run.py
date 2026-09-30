#!/usr/bin/env python3
"""Prepare and execute reproducible WAVE and native Metal experiments."""

import argparse
from array import array
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time

from workloads import ALL_WORKLOADS, WORKLOADS, assembly, make_case, native_source, repeated_expected


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_text(command):
    return subprocess.run(command, cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def checked(command, log, *, env=None, stdout=None):
    with log.open("ab") as errors:
        errors.write((json.dumps([str(part) for part in command]) + "\n").encode())
        errors.flush()
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout or errors, stderr=errors)
    if result.returncode:
        raise RuntimeError(f"command failed with status {result.returncode}; see {log}")


def source_manifest():
    result = {}
    for package in ("wave-asm", "wave-decode", "wave-emu", "wave-metal"):
        for path in (ROOT / package).rglob("*"):
            relative = path.relative_to(ROOT)
            if path.is_file() and "target" not in relative.parts and path.suffix in (".rs", ".toml", ".lock"):
                result[str(relative)] = digest(path)
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in (".py", ".swift", ".md"):
            result[str(path.relative_to(ROOT))] = digest(path)
    return dict(sorted(result.items()))


def hardware_metadata():
    profile = json.loads(command_text(["system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"]))
    machine = profile["SPHardwareDataType"][0]
    return {
        "machine": {key: machine[key] for key in (
            "chip_type", "machine_model", "machine_name", "model_number",
            "number_processors", "physical_memory", "boot_rom_version", "os_loader_version",
        ) if key in machine},
        "gpus": [{key: gpu[key] for key in (
            "sppci_model", "sppci_cores", "spdisplays_vendor", "spdisplays_metal", "sppci_bus",
        ) if key in gpu} for gpu in profile["SPDisplaysDataType"]],
    }


def prepare(args, output):
    if sys.byteorder != "little" or array("I").itemsize != 4:
        raise RuntimeError("the experiment requires a little-endian host with 32-bit unsigned words")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"state": "preparing"})
    sources = source_manifest()
    write_json(output / "source-manifest.json", sources)
    for relative in sources:
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    (output / "working-tree.patch").write_text(command_text(["git", "diff", "--binary", "HEAD"]))
    metadata = {
        "schema_version": 2, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_revision": command_text(["git", "rev-parse", "HEAD"]),
        "git_status": command_text(["git", "status", "--porcelain"]),
        "source_manifest_sha256": digest(output / "source-manifest.json"),
        "working_tree_patch_sha256": digest(output / "working-tree.patch"),
        "python": sys.version, "platform": platform.platform(),
        "swift": command_text(["swiftc", "--version"]),
        "cargo": command_text(["cargo", "--version"]),
        "rustc": command_text(["rustc", "--version"]),
        "hardware": hardware_metadata(),
        "sizes": args.sizes, "workloads": args.workloads, "warmup": args.warmup,
        "repeats": args.repeats, "input_seed": args.seed, "order_seed": args.order_seed,
        "dispatches_per_sample": args.dispatches_per_sample,
        "protocol": "single_dispatch" if args.dispatches_per_sample == 1 else "amortized_repeated_dispatch",
        "timing": {
            "gpu_batch_seconds": "Command buffer GPU end minus GPU start across the recorded dispatch count.",
            "host_batch_seconds": "Host elapsed time for encoding, submission and completion. Excludes memory reset and correctness validation.",
            "gpu_seconds": "GPU batch interval divided by dispatch count; includes encoder scheduling and tracked resource hazards.",
            "host_dispatch_seconds": "Host batch interval divided by dispatch count.",
            "compile_pipeline_seconds": "Metal library and pipeline construction. Driver cache state is uncontrolled.",
        },
        "numerics": "Finite, exactly representable inputs and sums; all allocation words must match the CPU oracle exactly.",
    }
    write_json(output / "environment.json", metadata)
    build = Path(args.build_dir).resolve()
    env = dict(os.environ, CARGO_TARGET_DIR=str(build))
    for package in ("wave-asm", "wave-metal", "wave-emu"):
        checked(["cargo", "build", "--offline", "--release", "--manifest-path", ROOT / package / "Cargo.toml"],
                output / "build.log", env=env)
    runner = output / "metal_runner"
    checked(["swiftc", "-O", "-warnings-as-errors", "-module-cache-path", build / "swift-module-cache",
             HERE / "metal_runner.swift", "-o", runner], output / "build.log")
    if source_manifest() != sources:
        raise RuntimeError("toolchain or experiment source changed during the build; prepare a fresh artifact")
    artifacts = {"metal_runner": digest(runner)}
    tool_hashes = {package: digest(build / "release" / package)
                   for package in ("wave-asm", "wave-metal", "wave-emu")}
    write_json(output / "tools.json", tool_hashes)
    configs = []
    for name in args.workloads:
        wave = output / f"{name}.wave"
        wbin = wave.with_suffix(".wbin")
        metal = output / f"{name}.wave.metal"
        native = output / f"{name}.native.metal"
        wave.write_text(assembly(name))
        native.write_text(native_source(name))
        checked([build / "release/wave-asm", wave, "-o", wbin], output / "build.log")
        checked([build / "release/wave-metal", wbin, "-o", metal], output / "build.log")
        for path in (wave, wbin, metal, native):
            artifacts[path.name] = digest(path)
        for size in args.sizes:
            print(f"Preparing {name}, n={size}", flush=True)
            initial, single_expected, case = make_case(name, size, args.seed)
            expected = repeated_expected(single_expected, case, args.dispatches_per_sample)
            prefix = output / f"{name}-{size}"
            initial_path = prefix.with_suffix(".input.bin")
            expected_path = prefix.with_suffix(".expected.bin")
            single_expected_path = prefix.with_suffix(".single-dispatch.expected.bin")
            initial_path.write_bytes(initial)
            expected_path.write_bytes(expected)
            single_expected_path.write_bytes(single_expected)
            case.update(initial_path=initial_path.name, expected_path=expected_path.name,
                        wave_metal_path=metal.name, native_metal_path=native.name,
                        wbin_path=wbin.name, single_dispatch_expected_path=single_expected_path.name)
            emulator_command = [build / "release/wave-emu", wbin,
                "--grid", f"{case['groups']},1,1", "--wave-width", "32",
                "--device-memory", str(len(initial)), "--arg", f"0:{initial_path}",
                "--max-instructions", "0", "--dump-u32", f"0:{len(initial) // 4}"]
            started = time.perf_counter()
            completed = subprocess.run(emulator_command, cwd=ROOT, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            prefix.with_suffix(".emulator.log").write_text(completed.stderr)
            if completed.returncode:
                raise RuntimeError(f"emulator failed for {name} at {size}: {completed.stderr}")
            actual = array("I", (int(line) for line in completed.stdout.splitlines())).tobytes()
            if actual != single_expected:
                actual_path = prefix.with_suffix(".emulator.actual.bin")
                actual_path.write_bytes(actual)
                raise RuntimeError(f"emulator disagrees with CPU oracle for {name} at {size}: {actual_path}")
            case["emulator"] = {"correct": True, "dispatches": 1, "wall_seconds": time.perf_counter() - started,
                                "memory_sha256": hashlib.sha256(actual).hexdigest()}
            for path in (initial_path, expected_path, single_expected_path):
                artifacts[path.name] = digest(path)
            configs.append(case)
    random.Random(args.order_seed).shuffle(configs)
    config = {"cases": configs, "warmup": args.warmup, "repeats": args.repeats,
              "order_seed": args.order_seed, "dispatches_per_sample": args.dispatches_per_sample}
    write_json(output / "config.json", config)
    artifacts["config.json"] = digest(output / "config.json")
    write_json(output / "artifacts.json", artifacts)
    write_json(output / "status.json", {"state": "prepared", "emulator_correct": True})


def percentile(values, quantile):
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summarize(output):
    records = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
    config = json.loads((output / "config.json").read_text())
    summary = []
    rng = random.Random(config["order_seed"])
    for case in config["cases"]:
        samples = [record for record in records if record["kind"] == "sample"
                   and record["workload"] == case["name"] and record["size"] == case["size"]
                   and not record["warmup"]]
        variants = {variant: sorted([r for r in samples if r["variant"] == variant],
                                    key=lambda r: r["repeat"]) for variant in ("wave", "native")}
        if any(len(rows) != config["repeats"] for rows in variants.values()):
            raise RuntimeError(f"incomplete samples for {case['name']} at {case['size']}")
        logs = [math.log(wave["gpu_seconds"] / native["gpu_seconds"])
                for wave, native in zip(variants["wave"], variants["native"])]
        bootstrap = [math.exp(statistics.mean(rng.choices(logs, k=len(logs)))) for _ in range(2000)]
        row = {"workload": case["name"], "size": case["size"], "correct": True,
               "dispatches_per_sample": config["dispatches_per_sample"],
               "wave_over_native_gpu_ratio": math.exp(statistics.mean(logs)),
               "within_run_paired_bootstrap_95_ci": [percentile(bootstrap, .025), percentile(bootstrap, .975)],
               "variants": {}}
        for variant, rows in variants.items():
            row["variants"][variant] = {
                field: {"median": statistics.median([r[field] for r in rows]),
                        "p05": percentile([r[field] for r in rows], .05),
                        "p95": percentile([r[field] for r in rows], .95)}
                for field in ("gpu_seconds", "host_dispatch_seconds")}
        summary.append(row)
    write_json(output / "summary.json", summary)


def execute(output):
    artifacts = json.loads((output / "artifacts.json").read_text())
    for relative, expected_hash in artifacts.items():
        if digest(output / relative) != expected_hash:
            raise RuntimeError(f"artifact changed after preparation: {relative}")
    sample_path = output / "samples.jsonl"
    if sample_path.exists():
        status = json.loads((output / "status.json").read_text())
        if status["state"] != "failed":
            raise RuntimeError("this artifact already contains a completed GPU run")
        suffix = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        sample_path.rename(output / f"samples.failed-{suffix}.jsonl")
        (output / "status.json").rename(output / f"status.failed-{suffix}.json")
    started = time.perf_counter()
    with sample_path.open("xb") as samples:
        checked([output / "metal_runner", output / "config.json"], output / "metal.log", stdout=samples)
    summarize(output)
    write_json(output / "status.json", {"state": "completed", "correct": True,
        "gpu_process_wall_seconds": time.perf_counter() - started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-prepared", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=[257, 16387, 262147])
    parser.add_argument("--workloads", choices=ALL_WORKLOADS, nargs="+", default=list(WORKLOADS))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--order-seed", type=int, default=20260930)
    parser.add_argument("--dispatches-per-sample", type=int, default=1)
    parser.add_argument("--build-dir", default="/private/tmp/wave-gpu-build")
    args = parser.parse_args()
    if args.run_prepared:
        if args.output or args.prepare_only:
            parser.error("--run-prepared cannot be combined with --output or --prepare-only")
        output = args.run_prepared.resolve()
    else:
        if not args.output:
            parser.error("--output is required unless --run-prepared is set")
        if any(size <= 0 or size > 100_000_000 for size in args.sizes):
            parser.error("sizes must be positive and at most 100000000")
        if len(args.sizes) != len(set(args.sizes)) or len(args.workloads) != len(set(args.workloads)):
            parser.error("sizes and workloads must not contain duplicates")
        if args.warmup < 0 or args.repeats <= 0 or args.dispatches_per_sample <= 0 or any(
            seed < 0 or seed > 2**64 - 1 for seed in (args.seed, args.order_seed)
        ):
            parser.error("invalid warmup, repeats, or seed")
        output = args.output.resolve()
        if output.exists():
            parser.error("output directory already exists")
        try:
            prepare(args, output)
        except Exception as error:
            if output.exists():
                write_json(output / "status.json", {
                    "state": "failed", "stage": "preparation", "error": str(error),
                })
            raise
    if not args.prepare_only:
        try:
            execute(output)
        except Exception as error:
            status_path = output / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get("state") != "completed":
                write_json(status_path, {"state": "failed", "error": str(error)})
            raise
    print(f"Experiment artifacts: {output}")


if __name__ == "__main__":
    main()
