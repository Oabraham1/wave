"""Run and verify separate NVIDIA process sessions on one GPU allocation."""

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
import run


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_json(data):
    def reject(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    return json.loads(data, parse_constant=reject)


def read_json(path):
    return parse_json(path.read_text())


def verify_artifacts(directory):
    manifest = read_json(directory / "artifacts.json")
    files = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()}
    require(set(manifest) | {"artifacts.json"} == files, "session artifact coverage differs")
    for name, expected in manifest.items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()), f"invalid artifact path: {name}")
        require(run.digest(path.read_bytes()) == expected, f"session checksum mismatch: {name}")


def verify_session(payload, directory, planned, parameters):
    """Validate a complete process record and recompute its summary from samples."""
    verify_artifacts(directory)
    require(read_json(directory / "status.json")["state"] == "completed", "session did not complete")
    manifest = run.verify_payload(payload)
    protocol = read_json(directory / "protocol.json")
    for key in ("repeats", "warmup", "dispatches_per_sample"):
        require(protocol[key] == parameters[key], f"session protocol differs: {key}")
    require(protocol["order_seed"] == planned["seed"], "session order seed differs")
    require(protocol["payload_manifest_sha256"] == run.digest((payload / "manifest.json").read_bytes()),
            "session used a different payload")
    require(read_json(directory / "payload-manifest.json") == manifest, "session payload manifest differs")
    cases = read_json(payload / "config.json")["cases"]
    keys = [(case["name"], case["size"]) for case in cases]
    require(cases and len(set(keys)) == len(keys), "missing or duplicate payload cases")
    dispatches = parameters["dispatches_per_sample"]
    references, repeated = {}, {}
    for case, key in zip(cases, keys):
        require(case["comparison"] == "bit_exact", "unsupported comparison contract")
        expected = (payload / "data" / case["expected_path"]).read_bytes()
        references[key] = run.digest(expected)
        repeated[key] = run.digest(run.expected_for_dispatches(expected, case, dispatches))
        require(case["emulator"]["correct"] is True and
                case["emulator"]["memory_sha256"] == references[key], "emulator reference differs")

    checks = read_json(directory / "correctness.json")
    expected_checks = {(name, size, variant) for name, size in keys for variant in ("wave", "native")}
    seen = set()
    for check in checks:
        key = (check["name"], check["size"])
        identity = (*key, check["variant"])
        require(identity in expected_checks and identity not in seen, "unexpected correctness record")
        seen.add(identity)
        require(check["correct"] is True and check["dispatches"] == 1 and
                check["memory_sha256"] == references[key], "single-dispatch correctness differs")
    require(seen == expected_checks, "single-dispatch correctness coverage differs")

    rng = random.Random(planned["seed"])
    ordered = list(keys)
    rng.shuffle(ordered)
    require(protocol["case_order"] == [list(key) for key in ordered], "session case order differs")
    expected_order = []
    for key in ordered:
        for pair in range(-parameters["warmup"], parameters["repeats"]):
            variants = ["wave", "native"]
            rng.shuffle(variants)
            expected_order.extend((*key, pair, variant, position)
                                  for position, variant in enumerate(variants))
    rows = [parse_json(line)
            for line in (directory / "samples.jsonl").read_text().splitlines()]
    require(len(rows) == len(expected_order), "sample coverage differs")
    for row, identity in zip(rows, expected_order):
        key = (row["name"], row["size"])
        require((*key, row["pair"], row["variant"], row["position"]) == identity,
                "sample identity or randomized order differs")
        require(row["warmup"] is (row["pair"] < 0) and row["dispatches"] == dispatches,
                "sample phase or dispatch count differs")
        require(row["correct"] is True and row["memory_sha256"] == repeated[key],
                "sample correctness differs")
        require(all(isinstance(row[field], (int, float)) and not isinstance(row[field], bool) and
                    math.isfinite(row[field]) and row[field] >= 0
                    for field in ("event_ms", "event_us_per_dispatch", "host_ns")),
                "invalid sample timing")
        require(row["host_ns"] > 0 and
                math.isclose(row["event_us_per_dispatch"], row["event_ms"] * 1000 / dispatches,
                             rel_tol=1e-12, abs_tol=1e-12), "sample timing units differ")
    compilation = read_json(directory / "compilation.json")
    require(compilation and all(command["returncode"] == 0 for command in compilation),
            "compiler record is missing or failed")
    run.verify_rounding_record(payload, directory)
    run.verify_disassembly(directory, sorted({name for name, _ in keys}))
    measured = [row for row in rows if not row["warmup"]]
    summary = run.summarize(measured, planned["seed"])
    require(read_json(directory / "summary.json") == summary, "stored summary differs from raw samples")
    return {"session": planned["session"], "seed": planned["seed"],
            "environment": read_json(directory / "environment.json"), "summary": summary,
            "samples_sha256": run.digest((directory / "samples.jsonl").read_bytes()),
            "artifacts_sha256": run.digest((directory / "artifacts.json").read_bytes())}


def summarize_sessions(sessions, expected_sessions, seed):
    """Give each completed process equal weight and expose incomplete campaigns."""
    grouped = {}
    for session in sessions:
        for row in session["summary"]:
            grouped.setdefault((row["name"], row["size"]), []).append((session["session"], row))
    rng = random.Random(seed)
    rows = []
    for (name, size), values in sorted(grouped.items()):
        item = {"name": name, "size": size, "verified_sessions": len(values),
                "expected_sessions": expected_sessions,
                "complete": len(values) == expected_sessions,
                "process_results": [{"session": session, **row} for session, row in values]}
        if all("wave_over_native_geomean" in row for _, row in values):
            ratios = [row["wave_over_native_geomean"] for _, row in values]
            logs = [math.log(value) for value in ratios]
            item.update({"wave_over_native_geomean": math.exp(statistics.mean(logs)),
                         "process_ratio_median": statistics.median(ratios),
                         "process_ratio_range": [min(ratios), max(ratios)]})
            if len(values) >= 2:
                bootstrap = sorted(math.exp(statistics.mean(rng.choices(logs, k=len(logs))))
                                   for _ in range(10000))
                item["between_process_bootstrap_95"] = [bootstrap[249], bootstrap[9749]]
        else:
            item["ratio_unavailable"] = "one or more event durations reached timer resolution"
        rows.append(item)
    return {"state": "completed" if len(sessions) == expected_sessions else "incomplete",
            "expected_sessions": expected_sessions, "verified_sessions": len(sessions),
            "sampling_unit": "separate process on the same GPU allocation",
            "statistics": "equal-weight geometric mean of process ratios; percentile bootstrap resamples processes",
            "interpretation": "describes process variation within this allocation; no equivalence test or inference across GPU allocations",
            "cases": rows}


def execute_campaign(payload, output, sessions, repeats, warmup, dispatches, seed):
    run.verify_payload(payload)
    payload_sha256 = run.digest((payload / "manifest.json").read_bytes())
    parameters = {"repeats": repeats, "warmup": warmup, "dispatches_per_sample": dispatches}
    seeds = random.Random(seed).sample(range(2**31), sessions)
    plan = [{"session": f"session-{index + 1:02d}", "seed": value}
            for index, value in enumerate(seeds)]
    run.write_json(output / "campaign.json", {
        "schema": 1, "seed": seed, "parameters": parameters, "sessions": plan,
        "payload_manifest_sha256": payload_sha256,
        "sampling_unit": "separate process on the same GPU allocation",
    })
    attempts, verified = [], []
    for planned in plan:
        directory = output / planned["session"]
        require(not directory.exists(), f"session directory already exists: {directory}")
        arguments = [sys.executable, "-B", str(payload / "run.py"), "--payload", str(payload),
                     "--output", str(directory), "--repeats", str(repeats), "--warmup", str(warmup),
                     "--dispatches", str(dispatches), "--seed", str(planned["seed"])]
        attempt = {**planned, "state": "running", "started_unix_seconds": time.time(),
                   "log": f"{planned['session']}.log", "argv": arguments}
        attempts.append(attempt)
        run.write_json(output / "attempts.json", attempts)
        try:
            require(run.digest((payload / "manifest.json").read_bytes()) == payload_sha256,
                    "payload manifest changed during campaign")
            with (output / attempt["log"]).open("w") as log:
                completed = subprocess.run(arguments, stdout=log, stderr=subprocess.STDOUT, check=False)
            attempt["returncode"] = completed.returncode
            require(completed.returncode == 0, f"session process exited with code {completed.returncode}")
            require(run.digest((payload / "manifest.json").read_bytes()) == payload_sha256,
                    "payload manifest changed during campaign")
            result = verify_session(payload, directory, planned, parameters)
            if verified:
                require(result["environment"]["device"] == verified[0]["environment"]["device"],
                        "device or driver changed between process sessions")
            verified.append(result)
            attempt["state"] = "verified"
        except Exception as error:
            attempt.update({"state": "failed", "error": str(error)})
            (output / f"{planned['session']}-verification-error.txt").write_text(traceback.format_exc())
        finally:
            attempt["finished_unix_seconds"] = time.time()
            run.write_json(output / "attempts.json", attempts)
            run.write_json(output / "session-summaries.json", verified)
            run.write_json(output / "summary.json", summarize_sessions(verified, sessions, seed))
        print(f"{planned['session']}: {attempt['state']}", flush=True)
    require(len(verified) == sessions, "campaign is incomplete; preserve all failed process records")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dispatches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    if not 1 <= args.sessions <= 1000 or args.repeats <= 0 or args.warmup < 0 or args.dispatches <= 0:
        parser.error("sessions must be 1..1000; repeats and dispatches positive; warmup nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    status = {"state": "running", "started_unix_seconds": time.time()}
    run.write_json(args.output / "status.json", status)
    try:
        execute_campaign(args.payload.resolve(), args.output.resolve(), args.sessions,
                         args.repeats, args.warmup, args.dispatches, args.seed)
        status["state"] = "completed"
    except Exception as error:
        status.update({"state": "failed", "error": str(error)})
        (args.output / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        status["finished_unix_seconds"] = time.time()
        run.write_json(args.output / "status.json", status)
        hashes = {str(path.relative_to(args.output)): run.digest(path.read_bytes())
                  for path in sorted(args.output.rglob("*")) if path.is_file()}
        run.write_json(args.output / "artifacts.json", hashes)


if __name__ == "__main__":
    main()
