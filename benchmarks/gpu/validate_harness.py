#!/usr/bin/env python3
"""Check rejection of invalid counts and incorrect GPU results using a recorded runner."""

from array import array
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from verify_artifacts import verify
from workloads import repeated_expected


def validate(artifact):
    artifact = artifact.resolve()
    verify(artifact)
    base = json.loads((artifact / "config.json").read_text())
    if base.get("dispatches_per_sample", 1) <= 1:
        raise ValueError("validation requires a completed repeated-dispatch session")
    runner = artifact / "metal_runner"
    outcomes = []
    with tempfile.TemporaryDirectory(prefix="wave-harness-validation-") as temporary:
        directory = Path(temporary)

        def configuration(name):
            config = copy.deepcopy(base)
            item = min((case for case in config["cases"] if case["name"] == name), key=lambda case: case["size"])
            for key in item:
                if key.endswith("_path"):
                    item[key] = str(artifact / item[key])
            config.update(cases=[item], warmup=0, repeats=1)
            return config, item

        def reject(label, config, message, invalid_variant=None):
            path = directory / f"{label}.json"
            path.write_text(json.dumps(config))
            result = subprocess.run([runner, path], text=True, capture_output=True, timeout=30)
            if result.returncode == 0 or message not in result.stderr:
                raise AssertionError(f"{label} was not rejected: {result.stdout} {result.stderr}")
            records = [json.loads(line) for line in result.stdout.splitlines()]
            if any(record["kind"] == "sample" and (
                invalid_variant is None or record["variant"] == invalid_variant
            ) for record in records):
                raise AssertionError(f"{label} emitted a successful record for an invalid result")
            outcomes.append({"case": label, "rejected": True, "reason": result.stderr.strip()})

        for count in (0, -1):
            config, _ = configuration("vector_add")
            config["dispatches_per_sample"] = count
            reject(f"dispatch-count-{count}", config, "repeats and dispatches must be positive")

        config, item = configuration("vector_add")
        native = Path(item["native_metal_path"]).read_text()
        expression = "output[i] = a[i] + as_type<float>(b[i]);"
        if native.count(expression) != 1:
            raise AssertionError("unexpected native vector kernel")
        path = directory / "wrong-output.metal"
        path.write_text(native.replace(expression, "output[i] = 0.0f;"))
        item["native_metal_path"] = str(path)
        reject("incorrect-gpu-output", config, "expected", invalid_variant="native")

        config, item = configuration("vector_add")
        initial = array("I")
        initial.frombytes(Path(item["initial_path"]).read_bytes())
        expected = array("I")
        expected.frombytes(Path(item["expected_path"]).read_bytes())
        input_word = initial[1] // 4
        initial[input_word] = 0x7FC00000
        expected[input_word] = 0x7FC00000
        for key, words in (("initial_path", initial), ("expected_path", expected)):
            path = directory / f"nonfinite-{key}.bin"
            path.write_bytes(words.tobytes())
            item[key] = str(path)
        reject("nonfinite-gpu-output", config, "non-finite output")

        config, item = configuration("histogram_skewed")
        item["expected_path"] = item["single_dispatch_expected_path"]
        reject("unscaled-histogram-reference", config, "expected")

    fixture = array("I", [1]).tobytes()
    case = {"name": "histogram_uniform", "output_word_offset": 0, "output_words": 1}
    if repeated_expected(fixture, case, 0xFFFFFFFF) != array("I", [0xFFFFFFFF]).tobytes():
        raise AssertionError("the largest valid histogram count was rejected")
    try:
        repeated_expected(fixture, case, 0x100000000)
    except ValueError:
        outcomes.append({"case": "histogram-overflow", "rejected": True})
    else:
        raise AssertionError("histogram overflow was not rejected")

    for line in (artifact / "samples.jsonl").read_text().splitlines():
        sample = json.loads(line)
        if sample["kind"] == "sample":
            if sample["gpu_seconds"] * sample["dispatches"] != sample["gpu_batch_seconds"]:
                raise AssertionError("GPU normalization differs from the recorded batch interval")
            if sample["host_dispatch_seconds"] * sample["dispatches"] != sample["host_batch_seconds"]:
                raise AssertionError("host normalization differs from the recorded batch interval")
    verify(artifact)
    return {"artifact": artifact.name, "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
            "raw_and_normalized_intervals_agree": True, "cases": outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.artifact)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(f"Harness validation complete: {args.output}")


if __name__ == "__main__":
    main()
