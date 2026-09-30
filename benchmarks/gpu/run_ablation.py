#!/usr/bin/env python3
"""Compare WAVE shuffle and shared-memory reductions in randomized pairs."""

import argparse
import datetime
import json
import math
import random
import shutil
import statistics
import subprocess
import time
from pathlib import Path

from run import HERE, digest, hardware_metadata, percentile, write_json
from verify_artifacts import verify, verify_manifest


VARIANTS = ("shuffle", "shared")


def prepare(args, output):
    source = args.source_artifact.resolve()
    verify(source)
    original_config = json.loads((source / "config.json").read_text())
    original = {(case["name"], case["size"]): case for case in original_config["cases"]}
    output.mkdir(parents=True, exist_ok=False)
    artifacts, source_hashes, cases = {}, {}, []
    write_json(output / "status.json", {"state": "preparing"})
    for relative in json.loads((source / "source-manifest.json").read_text()):
        destination = output / "source" / "translation" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "source" / relative, destination)
        source_hashes[f"translation/{relative}"] = digest(destination)
    for filename in ("run_ablation.py", "run.py", "verify_artifacts.py", "metal_runner.swift"):
        destination = output / "source" / "harness" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / filename, destination)
        source_hashes[f"harness/{filename}"] = digest(destination)
    write_json(output / "source-manifest.json", source_hashes)

    def retain(relative):
        destination = output / relative
        shutil.copyfile(source / relative, destination)
        artifacts[relative] = digest(destination)
        return relative

    for size in args.sizes:
        pair = [original[(f"reduce_{variant}", size)] for variant in VARIANTS]
        left, right = pair
        for field in ("groups", "threads", "output_word_offset", "output_words", "output_type"):
            if left[field] != right[field]:
                raise ValueError(f"reduction geometry differs for {field} at {size}")
        for field in ("initial_path", "expected_path"):
            if (source / left[field]).read_bytes() != (source / right[field]).read_bytes():
                raise ValueError(f"reduction data differ for {field} at {size}")
        single_path = left.get("single_dispatch_expected_path", left["expected_path"])
        if (source / left["expected_path"]).read_bytes() != (source / single_path).read_bytes():
            raise ValueError("reduction reference must be invariant under repeated dispatch")
        paths, binaries = {}, {}
        for variant, case in zip(VARIANTS, pair):
            paths[variant] = retain(case["wave_metal_path"])
            binaries[variant] = retain(case["wbin_path"])
            retain(f"reduce_{variant}.wave")
        item = {field: left[field] for field in (
            "size", "seed", "groups", "threads", "output_word_offset", "output_words", "output_type",
            "comparison", "absolute_tolerance", "relative_tolerance",
        )}
        item.update(name="reduction_communication", variant_metal_paths=paths,
                    variant_wbin_paths=binaries,
                    variant_entry_points={variant: f"reduce_{variant}" for variant in VARIANTS},
                    wave_metal_path=paths["shuffle"], native_metal_path=paths["shared"],
                    emulator={variant: case["emulator"] for variant, case in zip(VARIANTS, pair)})
        for field in ("initial_path", "expected_path"):
            item[field] = retain(left[field])
        item["single_dispatch_expected_path"] = retain(single_path)
        cases.append(item)
    random.Random(args.order_seed).shuffle(cases)
    config = {"cases": cases, "variant_names": list(VARIANTS), "warmup": args.warmup,
              "repeats": args.repeats, "order_seed": args.order_seed,
              "dispatches_per_sample": args.dispatches_per_sample}
    write_json(output / "config.json", config)
    artifacts["config.json"] = digest(output / "config.json")
    environment = {
        "schema_version": 1, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hardware": hardware_metadata(), "source_artifact_name": source.name,
        "source_config_sha256": digest(source / "config.json"),
        "source_artifacts_manifest_sha256": digest(source / "artifacts.json"),
        "source_manifest_sha256": digest(output / "source-manifest.json"),
        "comparison": "WAVE shared-memory XOR-tree reduction divided by WAVE shuffle XOR-tree reduction",
        "protocol": "single_dispatch" if args.dispatches_per_sample == 1 else "amortized_repeated_dispatch",
        "timing": "Metal command-buffer GPU interval divided by recorded dispatch count; host encoding, submission and completion recorded separately.",
        "reset": "Both variants start from identical original allocations before every sample.",
        "correctness": "Full allocation equality after every warmup and measured sample.",
        "algorithm": "Each wave reduces 32 values at XOR offsets 16, 8, 4, 2, 1; tails are zero padded. Shared memory uses matching partner indices and barriers.",
    }
    write_json(output / "environment.json", environment)
    subprocess.run(["swiftc", "-O", "-warnings-as-errors", "-module-cache-path",
                    "/private/tmp/wave-gpu-build/swift-module-cache",
                    output / "source/harness/metal_runner.swift", "-o", output / "metal_runner"], check=True)
    artifacts["metal_runner"] = digest(output / "metal_runner")
    write_json(output / "artifacts.json", artifacts)
    write_json(output / "status.json", {"state": "prepared"})


def summarize(output):
    config = json.loads((output / "config.json").read_text())
    records = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
    expected = {(case["name"], case["size"], variant, repeat)
                for case in config["cases"] for variant in VARIANTS
                for repeat in range(-config["warmup"], config["repeats"])}
    samples = {}
    for sample in records:
        if sample["kind"] != "sample":
            continue
        key = (sample["workload"], sample["size"], sample["variant"], sample["repeat"])
        if key not in expected or key in samples or sample["correct"] is not True or sample["gpu_status"] != "completed":
            raise ValueError(f"unexpected or incorrect sample: {key}")
        if sample["warmup"] != (sample["repeat"] < 0) or sample["dispatches"] != config["dispatches_per_sample"]:
            raise ValueError(f"sample protocol differs: {key}")
        for field in ("gpu_seconds", "gpu_batch_seconds", "host_dispatch_seconds", "host_batch_seconds"):
            if not math.isfinite(sample[field]) or sample[field] <= 0:
                raise ValueError(f"invalid timing: {key}")
        for normalized, raw in (("gpu_seconds", "gpu_batch_seconds"), ("host_dispatch_seconds", "host_batch_seconds")):
            if not math.isclose(sample[normalized] * sample["dispatches"], sample[raw], rel_tol=1e-12):
                raise ValueError(f"incorrect dispatch normalization: {key}")
        samples[key] = sample
    if samples.keys() != expected:
        raise ValueError("incomplete reduction samples")
    rng, summary = random.Random(config["order_seed"]), []
    for case in config["cases"]:
        paired, timings = [], {variant: [] for variant in VARIANTS}
        for repeat in range(-config["warmup"], config["repeats"]):
            pair = {variant: samples[(case["name"], case["size"], variant, repeat)] for variant in VARIANTS}
            if {record["order_in_pair"] for record in pair.values()} != {0, 1}:
                raise ValueError("pair order is incomplete")
            if repeat >= 0:
                paired.append(math.log(pair["shared"]["gpu_seconds"] / pair["shuffle"]["gpu_seconds"]))
                for variant in VARIANTS:
                    timings[variant].append(pair[variant]["gpu_seconds"])
        bootstrap = [math.exp(statistics.mean(rng.choices(paired, k=len(paired)))) for _ in range(2000)]
        summary.append({"workload": case["name"], "size": case["size"], "correct": True,
                        "dispatches_per_sample": config["dispatches_per_sample"],
                        "shared_over_shuffle_gpu_ratio": math.exp(statistics.mean(paired)),
                        "within_run_paired_bootstrap_95_ci": [percentile(bootstrap, .025), percentile(bootstrap, .975)],
                        "variants": {variant: {"gpu_seconds": {"median": statistics.median(values),
                                     "p05": percentile(values, .05), "p95": percentile(values, .95)}}
                                     for variant, values in timings.items()}})
    write_json(output / "summary.json", summary)


def execute(output):
    verify_manifest(output, json.loads((output / "artifacts.json").read_text()))
    verify_manifest(output / "source", json.loads((output / "source-manifest.json").read_text()))
    start = time.perf_counter()
    with (output / "samples.jsonl").open("xb") as samples, (output / "metal.log").open("xb") as log:
        subprocess.run([output / "metal_runner", output / "config.json"], stdout=samples, stderr=log, check=True)
    summarize(output)
    write_json(output / "status.json", {"state": "completed", "correct": True,
               "gpu_process_wall_seconds": time.perf_counter() - start})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-prepared", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--sizes", nargs="+", type=int, default=[257, 16387, 262147])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--dispatches-per-sample", type=int, default=32)
    parser.add_argument("--order-seed", type=int, default=20260950)
    args = parser.parse_args()
    if args.run_prepared:
        if args.output or args.source_artifact or args.prepare_only:
            parser.error("--run-prepared cannot be combined with preparation arguments")
        output = args.run_prepared.resolve()
    else:
        if not args.source_artifact or not args.output:
            parser.error("--source-artifact and --output are required")
        if args.warmup < 0 or args.repeats <= 0 or args.dispatches_per_sample <= 0 or not 0 <= args.order_seed < 2**64:
            parser.error("invalid warmup, repeats, dispatches, or seed")
        if any(size <= 0 for size in args.sizes) or len(args.sizes) != len(set(args.sizes)):
            parser.error("sizes must be unique and positive")
        output = args.output.resolve()
        prepare(args, output)
    if not args.prepare_only:
        try:
            execute(output)
        except Exception as error:
            write_json(output / "status.json", {"state": "failed", "error": str(error)})
            raise
    print(f"Reduction ablation artifacts: {output}")


if __name__ == "__main__":
    main()
