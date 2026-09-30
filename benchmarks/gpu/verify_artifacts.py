#!/usr/bin/env python3
"""Verify a completed session's file hashes and dispatch records."""

import argparse
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def verify_manifest(root, manifest):
    root = root.resolve()
    for relative, expected in manifest.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"path leaves artifact directory: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA256 mismatch: {path}")


def verify(root):
    if read_json(root / "status.json").get("state") != "completed":
        raise ValueError(f"session is incomplete: {root}")
    verify_manifest(root, read_json(root / "artifacts.json"))
    verify_manifest(root / "source", read_json(root / "source-manifest.json"))
    config = read_json(root / "config.json")
    expected = {
        (case["name"], case["size"], variant, repeat)
        for case in config["cases"] for variant in ("wave", "native")
        for repeat in range(-config["warmup"], config["repeats"])
    }
    actual = set()
    for line in (root / "samples.jsonl").read_text().splitlines():
        sample = json.loads(line)
        if sample["kind"] != "sample":
            continue
        key = (sample["workload"], sample["size"], sample["variant"], sample["repeat"])
        if key in actual or key not in expected:
            raise ValueError(f"duplicate or unexpected sample: {key}")
        actual.add(key)
        if sample["correct"] is not True or sample["gpu_status"] != "completed":
            raise ValueError(f"unsuccessful sample: {key}")
        if sample.get("dispatches", 1) != config.get("dispatches_per_sample", 1):
            raise ValueError(f"dispatch count differs: {key}")
        if sample["warmup"] != (sample["repeat"] < 0):
            raise ValueError(f"warmup marker differs: {key}")
    if actual != expected:
        raise ValueError(f"missing dispatch samples: {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", type=Path, nargs="+")
    args = parser.parse_args()
    for root in args.sessions:
        verify(root)
        print(f"Verified: {root}")


if __name__ == "__main__":
    main()
