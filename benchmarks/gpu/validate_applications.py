#!/usr/bin/env python3
"""Check that the GPU harness rejects application errors and guard corruption."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from verify_artifacts import verify


def validate(artifact):
    artifact = artifact.resolve()
    verify(artifact)
    base = json.loads((artifact / "config.json").read_text())
    outcomes = []
    mutations = [
        ("csr-empty-row", "csr_spmv", "output[i] = sum;",
         "if (row_ptr[i] != row_ptr[i + 1]) output[i] = sum;", "expected"),
        ("csr-column-index", "csr_spmv", "x[col_idx[j]]", "x[0]", "expected"),
        ("stencil-perimeter", "stencil_2d", "float value = input[i];", "float value = 0.0f;", "expected"),
        ("stencil-row-wrap", "stencil_2d", "x > 0 && x < width - 1 && y > 0 && y < height - 1",
         "y > 0 && y < height - 1", "expected"),
        ("stencil-nonfinite", "stencil_2d", "output[i] = value;",
         "output[i] = as_type<float>(0x7fc00000u);", "non-finite output"),
        ("csr-guard-write", "csr_spmv", "output[i] = sum;",
         "output[i] = sum; if (i == 0) memory[31] = 0;", "expected"),
    ]
    with tempfile.TemporaryDirectory(prefix="wave-application-faults-") as temporary:
        directory = Path(temporary)
        for label, name, original, mutated, diagnostic in mutations:
            config = copy.deepcopy(base)
            item = min((case for case in config["cases"] if case["name"] == name and case["size"] >= 257),
                       key=lambda case: case["size"])
            for key in item:
                if key.endswith("_path"):
                    item[key] = str(artifact / item[key])
            source = Path(item["native_metal_path"]).read_text()
            if source.count(original) != 1:
                raise AssertionError(f"unexpected native source for {label}")
            shader = directory / f"{label}.metal"
            shader.write_text(source.replace(original, mutated))
            item["native_metal_path"] = str(shader)
            config.update(cases=[item], warmup=0, repeats=1, dispatches_per_sample=1)
            config_path = directory / f"{label}.json"
            config_path.write_text(json.dumps(config))
            result = subprocess.run([artifact / "metal_runner", config_path], capture_output=True,
                                    text=True, timeout=60)
            if result.returncode == 0 or diagnostic not in result.stderr:
                raise AssertionError(f"{label} was not rejected: {result.stdout} {result.stderr}")
            records = [json.loads(line) for line in result.stdout.splitlines()]
            if any(record["kind"] == "sample" and record["variant"] == "native" for record in records):
                raise AssertionError(f"{label} emitted a successful incorrect result")
            outcomes.append({"case": label, "rejected": True, "diagnostic": result.stderr.strip(),
                             "mutated_shader_sha256": hashlib.sha256(shader.read_bytes()).hexdigest()})
    return {"artifact": str(artifact), "cases": outcomes,
            "runner_sha256": hashlib.sha256((artifact / "metal_runner").read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.artifact)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(f"Application fault checks complete: {args.output}")


if __name__ == "__main__":
    main()
