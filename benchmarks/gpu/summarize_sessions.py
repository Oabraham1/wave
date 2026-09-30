#!/usr/bin/env python3
"""Compare completed process runs without pooling their dispatch samples."""

import argparse
import json
from pathlib import Path
import statistics


def read_json(path):
    return json.loads(path.read_text())


def summarize(paths):
    sessions = []
    source_manifest = None
    inputs = None
    case_keys = None
    dispatches = None
    for path in paths:
        if read_json(path / "status.json").get("state") != "completed":
            raise ValueError(f"incomplete session: {path}")
        sources = read_json(path / "source-manifest.json")
        if source_manifest is not None and sources != source_manifest:
            raise ValueError(f"source snapshot differs: {path}")
        source_manifest = sources
        batch_count = read_json(path / "config.json").get("dispatches_per_sample", 1)
        if dispatches is not None and dispatches != batch_count:
            raise ValueError(f"dispatch protocol differs: {path}")
        dispatches = batch_count
        artifact_hashes = read_json(path / "artifacts.json")
        input_hashes = {key: value for key, value in artifact_hashes.items()
                        if key.endswith((".input.bin", ".expected.bin", ".wbin", ".metal"))}
        if inputs is not None and input_hashes != inputs:
            raise ValueError(f"inputs, expected results, or kernels differ: {path}")
        inputs = input_hashes
        rows = {(row["workload"], row["size"]): row for row in read_json(path / "summary.json")}
        if case_keys is not None and set(rows) != case_keys:
            raise ValueError(f"workloads differ: {path}")
        case_keys = set(rows)
        records = [json.loads(line) for line in (path / "samples.jsonl").read_text().splitlines()]
        sessions.append({"name": path.name, "rows": rows,
                         "max_thermal_state": max(record["thermal_state"] for record in records
                                                  if record["kind"] == "sample")})
    result = []
    for workload, size in sorted(case_keys):
        ratios = [session["rows"][(workload, size)]["wave_over_native_gpu_ratio"] for session in sessions]
        result.append({
            "workload": workload, "size": size,
            "median_session_wave_over_native_ratio": statistics.median(ratios),
            "session_ratio_range": [min(ratios), max(ratios)],
            "sessions": [{
                "name": session["name"], "wave_over_native_gpu_ratio": ratio,
                "max_thermal_state": session["max_thermal_state"],
                "wave_median_gpu_seconds": session["rows"][(workload, size)]["variants"]["wave"]["gpu_seconds"]["median"],
                "native_median_gpu_seconds": session["rows"][(workload, size)]["variants"]["native"]["gpu_seconds"]["median"],
            } for session, ratio in zip(sessions, ratios)],
        })
    return {"method": "Separate process runs; session ratios summarized without pooling dispatch samples.",
            "dispatches_per_sample": dispatches,
            "protocol": "single_dispatch" if dispatches == 1 else "amortized_repeated_dispatch",
            "results": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    args = parser.parse_args()
    if len(args.sessions) != len({path.resolve() for path in args.sessions}):
        parser.error("each session directory must be distinct")
    print(json.dumps(summarize(args.sessions), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
