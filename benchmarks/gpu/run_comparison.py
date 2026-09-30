#!/usr/bin/env python3
"""Compare WAVE, native Metal and Slang in repeated GPU process sessions."""

import argparse
import copy
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import time

import run
import slang_baseline
from verify_artifacts import verify_manifest
from workloads import ALL_WORKLOADS

VARIANTS = ("wave", "native", "slang")
PAIRS = (("wave", "native"), ("wave", "slang"), ("slang", "native"))


def read_json(path):
    return json.loads(path.read_text())


def sample_records(config, records):
    """Require a complete, correctly paired set of successful finite samples."""
    variants = config["variant_names"]
    expected = {
        (case["name"], case["size"], variant, repeat)
        for case in config["cases"] for variant in variants
        for repeat in range(-config["warmup"], config["repeats"])
    }
    samples = {}
    orders = {}
    for row in records:
        if row["kind"] != "sample":
            continue
        key = (row["workload"], row["size"], row["variant"], row["repeat"])
        if key in samples or key not in expected:
            raise ValueError(f"duplicate or unexpected sample: {key}")
        if row["correct"] is not True or row["gpu_status"] != "completed":
            raise ValueError(f"unsuccessful sample: {key}")
        if row["warmup"] != (row["repeat"] < 0) or row["dispatches"] != config["dispatches_per_sample"]:
            raise ValueError(f"sample protocol differs: {key}")
        for normalized, batch in (("gpu_seconds", "gpu_batch_seconds"),
                                  ("host_dispatch_seconds", "host_batch_seconds")):
            if any(not math.isfinite(row[field]) or row[field] <= 0 for field in (normalized, batch)):
                raise ValueError(f"nonpositive or non-finite time: {key}")
            if not math.isclose(row[normalized] * row["dispatches"], row[batch], rel_tol=1e-12):
                raise ValueError(f"time normalization differs: {key}")
        order_key = (row["workload"], row["size"], row["repeat"])
        orders.setdefault(order_key, []).append(row["order_in_pair"])
        samples[key] = row
    if set(samples) != expected:
        raise ValueError("missing samples")
    if any(sorted(order) != list(range(len(variants))) for order in orders.values()):
        raise ValueError("variant order is not a permutation")
    return samples


def session_summary(config, records):
    samples = sample_records(config, records)
    rng = random.Random(config["order_seed"])
    result = []
    for case in sorted(config["cases"], key=lambda item: (item["name"], item["size"])):
        name, size = case["name"], case["size"]
        values = {variant: [samples[(name, size, variant, repeat)]["gpu_seconds"]
                            for repeat in range(config["repeats"])] for variant in VARIANTS}
        ratios = {}
        for numerator, denominator in PAIRS:
            logs = [math.log(a / b) for a, b in zip(values[numerator], values[denominator])]
            bootstrap = [math.exp(statistics.mean(rng.choices(logs, k=len(logs)))) for _ in range(2000)]
            ratios[f"{numerator}_over_{denominator}"] = {
                "geomean": math.exp(statistics.mean(logs)),
                "within_process_bootstrap_95": [run.percentile(bootstrap, .025),
                                                 run.percentile(bootstrap, .975)],
            }
        result.append({"workload": name, "size": size, "ratios": ratios,
                       "median_gpu_seconds": {key: statistics.median(value) for key, value in values.items()},
                       "correct": True})
    return result


def campaign_summary(root, require_complete=True):
    """Recompute every statistic from the recorded process samples."""
    root = Path(root)
    plan = read_json(root / "plan.json")
    if require_complete and read_json(root / "status.json")["state"] != "completed":
        raise ValueError("campaign is incomplete")
    base = root / "base"
    verify_manifest(base, read_json(base / "artifacts.json"))
    verify_manifest(base / "source", read_json(base / "source-manifest.json"))
    session_maps = []
    base_config = read_json(base / "config.json")
    for index in range(plan["sessions"]):
        directory = root / f"session-{index + 1}"
        if read_json(directory / "status.json")["state"] != "completed":
            raise ValueError(f"incomplete process session: {directory.name}")
        config = read_json(directory / "config.json")
        wanted = session_config(base_config, plan, index)
        if config != wanted:
            raise ValueError(f"session configuration differs: {directory.name}")
        records = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
        summary = session_summary(config, records)
        if read_json(directory / "summary.json") != summary:
            raise ValueError(f"session summary differs from raw samples: {directory.name}")
        session_maps.append({(item["workload"], item["size"]): item for item in summary})
    rng = random.Random(plan["order_seed"])
    result = []
    for key in sorted(session_maps[0]):
        rows = [session[key] for session in session_maps]
        ratios = {}
        for numerator, denominator in PAIRS:
            label = f"{numerator}_over_{denominator}"
            values = [row["ratios"][label]["geomean"] for row in rows]
            logs = [math.log(value) for value in values]
            bootstrap = [math.exp(statistics.mean(rng.choices(logs, k=len(logs)))) for _ in range(5000)]
            ratios[label] = {"process_ratios": values, "geomean": math.exp(statistics.mean(logs)),
                             "between_process_bootstrap_95": [run.percentile(bootstrap, .025),
                                                              run.percentile(bootstrap, .975)]}
        result.append({"workload": key[0], "size": key[1], "ratios": ratios,
                       "process_median_gpu_seconds": [row["median_gpu_seconds"] for row in rows],
                       "correct": True})
    return {"plan": plan, "cases": result, "correct": True}


def session_config(base_config, plan, index):
    config = copy.deepcopy(base_config)
    seed = plan["order_seed"] + index
    config.update(order_seed=seed, warmup=plan["warmup"], repeats=plan["repeats"],
                  variant_names=list(VARIANTS))
    for case in config["cases"]:
        for field, value in list(case.items()):
            if field.endswith("_path"):
                case[field] = "../base/" + value
        case["variant_metal_paths"] = {"slang": "../base/" + case["name"] + ".slang.metal"}
    config["cases"].sort(key=lambda case: (case["name"], case["size"]))
    random.Random(seed).shuffle(config["cases"])
    return config


def prepare(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        "sessions": args.sessions, "sizes": args.sizes, "workloads": args.workloads,
        "warmup": args.warmup, "repeats": args.repeats, "input_seed": args.seed,
        "order_seed": args.order_seed, "dispatches_per_sample": args.dispatches_per_sample,
        "variants": list(VARIANTS), "slang_version": slang_baseline.VERSION,
        "session_unit": "fresh sequential Metal runner process on one Apple GPU",
        "statistics": "paired geometric time ratios within each process; equal-weight process means and percentile bootstrap over processes",
        "timing": "per-dispatch command-buffer GPU span over repeated dispatches, with full validation after each sample",
        "comparisons": "same algorithms, allocations, inputs, references, workgroup sizes and Metal math options",
        "stopping": "fixed process and sample counts; retain failures; no significance-based stopping",
    }
    run.write_json(root / "plan.json", plan)
    run.write_json(root / "status.json", {"state": "preparing"})
    base = root / "base"
    try:
        run.prepare(args, base)
        compiles = [slang_baseline.compile_source(name, base, args.slangc) for name in args.workloads]
        compiler = Path(args.slangc).resolve()
        libraries = sorted((compiler.parent.parent / "lib").glob("*slang*.dylib"))
        run.write_json(base / "slang-compiler.json", {
            "version": slang_baseline.VERSION, "release_url": slang_baseline.RELEASE_URL,
            "release_sha256": slang_baseline.RELEASE_SHA256,
            "compiler_sha256": run.digest(compiler),
            "libraries": {path.name: run.digest(path) for path in libraries}, "compiles": compiles,
        })
        artifacts = read_json(base / "artifacts.json")
        for path in itertools.chain(base.glob("*.slang*"), [base / "slang-compiler.json"]):
            artifacts[path.name] = run.digest(path)
        run.write_json(base / "artifacts.json", artifacts)
        run.write_json(root / "status.json", {"state": "prepared"})
    except Exception as error:
        run.write_json(root / "status.json", {"state": "failed", "stage": "preparation", "error": str(error)})
        raise


def execute(root):
    root = Path(root).resolve()
    if read_json(root / "status.json")["state"] != "prepared":
        raise ValueError("execution requires a fresh prepared campaign")
    plan = read_json(root / "plan.json")
    base = root / "base"
    verify_manifest(base, read_json(base / "artifacts.json"))
    verify_manifest(base / "source", read_json(base / "source-manifest.json"))
    config = read_json(base / "config.json")
    run.write_json(root / "status.json", {"state": "running"})
    try:
        for index in range(plan["sessions"]):
            directory = root / f"session-{index + 1}"
            directory.mkdir()
            run.write_json(directory / "config.json", session_config(config, plan, index))
            run.write_json(directory / "status.json", {"state": "running"})
            print(f"Running process {index + 1}/{plan['sessions']}", flush=True)
            started = time.perf_counter()
            with (directory / "samples.jsonl").open("xb") as output, (directory / "metal.log").open("xb") as errors:
                result = subprocess.run([base / "metal_runner", directory / "config.json"], stdout=output, stderr=errors)
            if result.returncode:
                run.write_json(directory / "status.json", {"state": "failed", "returncode": result.returncode})
                raise RuntimeError(f"GPU process failed; see {directory / 'metal.log'}")
            records = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
            run.write_json(directory / "summary.json", session_summary(read_json(directory / "config.json"), records))
            run.write_json(directory / "status.json", {"state": "completed", "correct": True,
                                                        "wall_seconds": time.perf_counter() - started})
        summary = campaign_summary(root, require_complete=False)
        run.write_json(root / "summary.json", summary)
        run.write_json(root / "status.json", {"state": "completed", "correct": True})
        hashes = {str(path.relative_to(root)): run.digest(path) for path in sorted(root.rglob("*"))
                  if path.is_file() and path.name != "record-manifest.json"}
        run.write_json(root / "record-manifest.json", hashes)
    except Exception as error:
        run.write_json(root / "status.json", {"state": "failed", "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-prepared", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--slangc", type=Path)
    parser.add_argument("--workloads", nargs="+", choices=ALL_WORKLOADS, default=list(ALL_WORKLOADS))
    parser.add_argument("--sizes", type=int, nargs="+", default=[257, 16387, 262147])
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--order-seed", type=int, default=20261001)
    parser.add_argument("--dispatches-per-sample", type=int, default=32)
    parser.add_argument("--build-dir", default="/private/tmp/wave-gpu-build")
    args = parser.parse_args()
    if args.verify:
        root = args.verify.resolve()
        verify_manifest(root, read_json(root / "record-manifest.json"))
        actual = campaign_summary(root)
        if actual != read_json(root / "summary.json"):
            raise ValueError("recorded campaign summary differs from raw samples")
        print("Campaign hashes, samples and recomputed statistics match.")
        return
    if args.run_prepared:
        if args.output or args.prepare_only:
            parser.error("--run-prepared cannot be combined with output or preparation")
        execute(args.run_prepared)
        return
    if not args.output or not args.slangc:
        parser.error("--output and --slangc are required")
    if args.sessions < 2 or args.warmup < 0 or args.repeats < 1 or args.dispatches_per_sample < 1:
        parser.error("need at least two sessions, nonnegative warmup and positive sample and dispatch counts")
    if any(size <= 0 or size > 100_000_000 for size in args.sizes):
        parser.error("invalid workload size")
    if len(set(args.sizes)) != len(args.sizes) or len(set(args.workloads)) != len(args.workloads):
        parser.error("duplicate workload or size")
    if args.seed < 0 or args.order_seed < 0 or args.order_seed + args.sessions > 2**64:
        parser.error("invalid seed")
    prepare(args)
    if not args.prepare_only:
        execute(args.output)


if __name__ == "__main__":
    main()
