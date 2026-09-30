#!/usr/bin/env python3
"""Record source-to-device checks and host launch timings on Apple hardware."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def sources():
    entries = {}
    for package in ("wave-runtime", "wave-compiler", "wave-asm", "wave-decode", "wave-emu",
                    "wave-metal", "wave-ptx", "wave-hip", "wave-sycl"):
        for path in (ROOT / package).rglob("*"):
            relative = path.relative_to(ROOT)
            if path.is_file() and "target" not in relative.parts and path.suffix in (".rs", ".swift", ".toml", ".lock"):
                entries[str(relative)] = path.read_bytes()
    entries[str(Path(__file__).resolve().relative_to(ROOT))] = Path(__file__).read_bytes()
    return entries


def capture(command):
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = sources()
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in sorted(snapshot.items())}
    (args.output / "source-manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    with (args.output / "source.tar.gz").open("wb") as file:
        with gzip.GzipFile(fileobj=file, mode="wb", filename="", mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for name, content in sorted(snapshot.items()):
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    info.mode = 0o644
                    archive.addfile(info, io.BytesIO(content))
    status = {"state": "running"}
    (args.output / "status.json").write_text(json.dumps(status) + "\n")
    try:
        build = ["cargo", "build", "--release", "--offline", "--manifest-path",
                 "wave-runtime/Cargo.toml", "--examples"]
        with (args.output / "build.log").open("w") as log:
            subprocess.run(build, cwd=ROOT, stdout=log, stderr=log, check=True)
        if sources() != snapshot:
            raise RuntimeError("source changed during build")
        environment = {
            "git_revision": capture(["git", "rev-parse", "HEAD"]),
            "rustc": capture(["rustc", "--version"]),
            "swift": capture(["swiftc", "--version"]),
            "os": capture(["sw_vers"]),
            "protocol": "Five sequential launches per size in one process; includes per-launch compilation and file transfers; no warmup exclusions.",
            "build_command": build,
        }
        (args.output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
        for example, filename, arguments in [("metal_validation", "samples.jsonl", ["5"]),
                                             ("metal_semantics", "semantics.jsonl", [])]:
            executable = ROOT / "wave-runtime" / "target" / "release" / "examples" / example
            environment[example + "_sha256"] = hashlib.sha256(executable.read_bytes()).hexdigest()
            with (args.output / filename).open("w") as output, (args.output / (example + ".stderr")).open("w") as errors:
                subprocess.run([str(executable), *arguments], cwd=ROOT, stdout=output, stderr=errors, check=True)
            rows = [json.loads(line) for line in (args.output / filename).read_text().splitlines()]
            if not rows or not all(row["correct"] for row in rows):
                raise RuntimeError(f"{example} did not report complete validation")
        if sources() != snapshot:
            raise RuntimeError("source changed during execution")
        (args.output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
        status = {"state": "completed"}
    except Exception as error:
        status = {"state": "failed", "error": str(error)}
        raise
    finally:
        (args.output / "status.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
