"""Package measured WBIN inputs and current PTX into a self-contained notebook."""

import argparse
import base64
import gzip
import hashlib
import io
import json
import lzma
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import textwrap

import rounding

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run(arguments):
    return subprocess.run(arguments, cwd=REPOSITORY, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()


def archive_bytes(root):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                contents = path.read_bytes()
                info = tarfile.TarInfo(str(path.relative_to(root)))
                info.size = len(contents)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(contents))
    return gzip.compress(raw.getvalue(), compresslevel=9, mtime=0)


def cell(kind, source):
    result = {"cell_type": kind, "id": digest(source.encode())[:12],
              "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        compile(source, "notebook-cell", "exec")
        result.update({"execution_count": None, "outputs": []})
    return result


def notebook(payload):
    encoded = base64.b64encode(payload).decode()
    extraction = '''import base64, hashlib, io, json, pathlib, tarfile, tempfile

PAYLOAD = (
ENCODED_PAYLOAD
)
compressed = base64.b64decode(PAYLOAD)
if hashlib.sha256(compressed).hexdigest() != "PAYLOAD_HASH":
    raise ValueError("embedded payload checksum mismatch")
workspace = pathlib.Path(tempfile.mkdtemp(prefix="wave-nvidia-"))
payload = workspace / "payload"
payload.mkdir()
with tarfile.open(fileobj=io.BytesIO(compressed), mode="r:*") as archive:
    for member in archive.getmembers():
        destination = (payload / member.name).resolve()
        if not member.isfile() or not destination.is_relative_to(payload.resolve()):
            raise ValueError(f"invalid payload entry: {member.name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.extractfile(member).read())
manifest = json.loads((payload / "manifest.json").read_text())
for name, expected in manifest.items():
    if hashlib.sha256((payload / name).read_bytes()).hexdigest() != expected:
        raise ValueError(f"payload checksum mismatch: {name}")
print("Verified frozen input, reference, source and PTX checksums.")
print(json.dumps(json.loads((payload / "origin.json").read_text()), indent=2))
'''.replace("ENCODED_PAYLOAD", "\n".join(f'    "{line}"' for line in textwrap.wrap(encoded, 100)))
    extraction = extraction.replace("PAYLOAD_HASH", digest(payload))
    cells = [
        cell("markdown", """# WAVE NVIDIA validation

Choose **Runtime → Change runtime type → GPU**, then run the cells in order.
This notebook contains the same WBIN, input and reference bytes used in the
accompanying Apple GPU session. No Drive mount, repository login or package
installation is needed. The runtime must provide an NVIDIA GPU with compute
capability 7.5 or newer, `nvcc`, `ptxas`, `nvdisasm` and `libcuda.so.1`.

The payload includes current WAVE PTX output, the backend source snapshot,
native CUDA source and checksums. NVIDIA `ptxas` compiles both PTX variants
for the allocated GPU. Every case checks the complete allocation against the
frozen emulator-verified reference, including input and guard words.
Before workload timing, each process checks a literal f32 rounding witness on
both WAVE and native CUDA. Separate multiplication and addition must produce
positive zero; explicit fused multiply-add must produce `0xa8800000`.
The record includes required disassembly for every compiled kernel.

CUDA event spans enclose kernel submissions. They can include host submission
gaps and are not pure instruction execution times. Transfers, compilation,
module loading and validation are outside the timed region. A separate host
span covers submission and synchronization. Results apply to this session
and workload subset. They do not establish full ISA conformance.
"""),
        cell("code", extraction),
        cell("markdown", """## Run the campaign

The default runs five separate Python processes on this GPU allocation. Each
process uses one dispatch per sample, five warmup pairs and forty measured
pairs per case. Distinct recorded seeds randomize case and variant order.
All warmup and measured outputs are checked against the same frozen references.

A process restart does not allocate a new GPU. The campaign measures variation
between processes within this allocation. Run the notebook on another GPU
allocation to collect a separate campaign and download each archive.

Changing `DISPATCHES` changes the experiment. Event spans then include a sequence
of host submissions, and histogram references account for accumulation.
A failed process is retained while later planned processes continue. Campaign
completion requires every planned process to pass. Repeating this cell creates
a fresh campaign and preserves earlier attempts in the download.
"""),
        cell("code", '''import subprocess, sys, uuid

SESSIONS = 5
REPEATS = 40
WARMUP = 5
DISPATCHES = 1
SEED = 20260930
results = workspace / ("campaign-" + uuid.uuid4().hex)
completed = subprocess.run([
    sys.executable, "-B", str(payload / "campaign.py"), "--payload", str(payload),
    "--output", str(results), "--repeats", str(REPEATS),
    "--sessions", str(SESSIONS),
    "--warmup", str(WARMUP), "--dispatches", str(DISPATCHES), "--seed", str(SEED),
])
print("Runner exit code:", completed.returncode)
print("Results:", results)
status_path = results / "status.json"
if status_path.is_file():
    print(status_path.read_text())
if completed.returncode == 0:
    print((results / "summary.json").read_text())
else:
    failure_path = results / "failure.txt"
    if failure_path.is_file():
        print(failure_path.read_text())
    else:
        print("Runner failed before writing an error record. See the process output above.")
'''),
        cell("markdown", """## Download the record

The archive contains raw samples, correctness records, device and compiler
metadata, compiler logs, executable kernels and the complete frozen payload.
Download it before ending the temporary runtime. Review `status.json` before
using any numbers. Per-process intervals resample measured pairs. Campaign
intervals resample process summaries with equal weight. Neither is an
equivalence test, and neither estimates variation across GPU allocations.
"""),
        cell("code", '''import shutil
from google.colab import files

archive = shutil.make_archive(str(workspace.with_name(workspace.name + "-record")),
                              "zip", root_dir=workspace)
print("Record SHA-256:", hashlib.sha256(pathlib.Path(archive).read_bytes()).hexdigest())
files.download(archive)
'''),
        cell("markdown", """## Sources

- [Colab runtime and resource limits](https://research.google.com/colaboratory/faq.html)
- [CUDA Driver API execution control](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EXEC.html)
- [CUDA event management](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EVENT.html)
- [NVIDIA compiler options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)

The notebook requires no credentials and does not publish results.
"""),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                        "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                        "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def measured_session(session):
    """Verify an Apple campaign and select its unchanged single-dispatch references."""
    base = session
    campaign_metadata = {}
    if (session / "base").is_dir():
        run([sys.executable, str(HERE.parent / "run_comparison.py"), "--verify", str(session)])
        base = session / "base"
        campaign_metadata["apple_campaign_record_manifest_sha256"] = digest(
            (session / "record-manifest.json").read_bytes())
    config_bytes = (base / "config.json").read_bytes()
    config = json.loads(config_bytes)
    measured = json.loads((base / "artifacts.json").read_text())
    if digest(config_bytes) != measured["config.json"]:
        raise ValueError("measured configuration checksum differs")
    if not config["cases"]:
        raise ValueError("measured configuration has no cases")
    for case in config["cases"]:
        case["expected_path"] = case.get("single_dispatch_expected_path", case["expected_path"])
    config["dispatches_per_sample"] = 1
    return base, config_bytes, config, measured, campaign_metadata


def prepare_rounding(payload, backend):
    """Assemble and emulate a literal witness outside the timed workloads."""
    data = payload / "data"
    source = data / f"{rounding.NAME}.wave"
    binary = data / rounding.CONTRACT["wbin_path"]
    initial = data / rounding.CONTRACT["initial_path"]
    expected = data / rounding.CONTRACT["expected_path"]
    source.write_text(rounding.SOURCE)
    initial.write_bytes(rounding.initial_bytes())
    expected.write_bytes(rounding.expected_bytes())
    assembler = REPOSITORY / "wave-asm/target/release/wave-asm"
    emulator = REPOSITORY / "wave-emu/target/release/wave-emu"
    run([str(assembler), str(source), "-o", str(binary)])
    command = [str(emulator), str(binary), "--grid", "1,1,1", "--wave-width", "32",
               "--device-memory", "32", "--arg", f"0:{initial}", "--max-instructions", "0",
               "--dump-u32", "0:8"]
    completed = subprocess.run(command, cwd=REPOSITORY, check=False, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (data / f"{rounding.NAME}.emulator.log").write_text(completed.stderr)
    if completed.returncode:
        raise ValueError(f"rounding witness emulator failed: {completed.stderr}")
    words = [int(line) for line in completed.stdout.splitlines()]
    if words != rounding.EXPECTED_WORDS:
        raise ValueError(f"rounding witness emulator output differs: {words}")
    actual = struct.pack("<8I", *words)
    (data / f"{rounding.NAME}.emulator.actual.bin").write_bytes(actual)
    run([str(backend), str(binary), "--sm", "75", "-o", str(data / rounding.CONTRACT["ptx_path"])])
    write_json(payload / "rounding-check.json", {
        **rounding.CONTRACT,
        "initial_sha256": digest(initial.read_bytes()), "expected_sha256": digest(expected.read_bytes()),
        "wbin_sha256": digest(binary.read_bytes()),
        "ptx_sha256": digest((data / rounding.CONTRACT["ptx_path"]).read_bytes()),
        "emulator": {"correct": True, "dispatches": 1, "memory_sha256": digest(actual)},
        "tools": {"assembler_sha256": digest(assembler.read_bytes()),
                  "emulator_sha256": digest(emulator.read_bytes()),
                  "ptx_backend_sha256": digest(backend.read_bytes())},
        "commands": [
            ["wave-asm", f"data/{source.name}", "-o", f"data/{binary.name}"],
            ["wave-emu", f"data/{binary.name}", "--grid", "1,1,1", "--wave-width", "32",
             "--device-memory", "32", "--arg", f"0:data/{initial.name}",
             "--max-instructions", "0", "--dump-u32", "0:8"],
            ["wave-ptx", f"data/{binary.name}", "--sm", "75", "-o", f"data/{rounding.CONTRACT['ptx_path']}"],
        ],
    })


def prepare(session, destination):
    base, config_bytes, config, measured, campaign_metadata = measured_session(session)
    for crate in ("wave-ptx", "wave-asm", "wave-emu"):
        run(["cargo", "build", "--release", "--locked", "--offline",
             "--manifest-path", f"{crate}/Cargo.toml"])
    backend = REPOSITORY / "wave-ptx/target/release/wave-ptx"
    with tempfile.TemporaryDirectory(prefix="wave-nvidia-package-") as temporary:
        payload = Path(temporary)
        data = payload / "data"
        data.mkdir()
        names = set()
        copied = set()
        for case in config["cases"]:
            if case["threads"] != 128 or case["comparison"] != "bit_exact":
                raise ValueError("unsupported launch or comparison contract")
            if not case.get("emulator", {}).get("correct"):
                raise ValueError("a case lacks successful emulator verification")
            names.add(case["name"])
            for key in ("initial_path", "expected_path", "wbin_path"):
                name = case[key]
                if Path(name).name != name:
                    raise ValueError(f"expected a flat measured artifact path: {name}")
                contents = (base / name).read_bytes()
                if digest(contents) != measured[name]:
                    raise ValueError(f"measured artifact checksum differs: {name}")
                (data / name).write_bytes(contents)
                copied.add(name)
            expected = (data / case["expected_path"]).read_bytes()
            if digest(expected) != case["emulator"]["memory_sha256"]:
                raise ValueError("reference differs from the recorded emulator output")
        commands = []
        for name in sorted(names):
            arguments = [str(backend), str(data / f"{name}.wbin"), "--sm", "75",
                         "-o", str(data / f"{name}.wave.ptx")]
            run(arguments)
            commands.append(["wave-ptx", f"data/{name}.wbin", "--sm", "75",
                             "-o", f"data/{name}.wave.ptx"])
            assembly = base / f"{name}.wave"
            if digest(assembly.read_bytes()) != measured[assembly.name]:
                raise ValueError(f"measured assembly checksum differs: {assembly.name}")
            shutil.copyfile(assembly, data / assembly.name)
        prepare_rounding(payload, backend)
        for name in ("run.py", "campaign.py", "native.cu", "prepare.py", "rounding.py", "rounding.cu"):
            shutil.copyfile(HERE / name, payload / name)
        for crate in ("wave-ptx", "wave-decode", "wave-asm", "wave-emu"):
            files = sorted((REPOSITORY / crate / "src").rglob("*.rs"))
            files += [REPOSITORY / crate / "Cargo.toml", REPOSITORY / crate / "Cargo.lock"]
            for source in files:
                target = payload / "source" / source.relative_to(REPOSITORY)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        shutil.copyfile(REPOSITORY / "LICENSE", payload / "source/LICENSE")
        write_json(payload / "config.json", config)
        (payload / "measured-config.json").write_bytes(config_bytes)
        shutil.copyfile(base / "artifacts.json", payload / "apple-base-artifacts.json")
        if campaign_metadata:
            shutil.copyfile(session / "record-manifest.json", payload / "apple-record-manifest.json")
        write_json(payload / "measured-artifacts.json", {name: measured[name] for name in sorted(copied)})
        write_json(payload / "origin.json", {
            "repository": "https://github.com/Oabraham1/wave",
            "head_commit": run(["git", "rev-parse", "HEAD"]),
            "source_identity": "the included source snapshot and manifest identify the working tree used",
            "measured_session": str(session.relative_to(REPOSITORY)) if session.is_relative_to(REPOSITORY) else session.name,
            "measured_config_sha256": digest(config_bytes),
            "payload_config_sha256": digest((payload / "config.json").read_bytes()),
            "reference_selection": "expected_path selects the unchanged Apple single-dispatch reference; payload dispatches_per_sample is 1",
            **campaign_metadata,
            "ptx_backend_executable_sha256": digest(backend.read_bytes()),
            "rustc": run(["rustc", "--version"]), "cargo": run(["cargo", "--version"]),
            "ptx_generation_commands": commands,
            "nvidia_execution": "not performed during notebook preparation",
        })
        files = {str(path.relative_to(payload)): digest(path.read_bytes())
                 for path in sorted(payload.rglob("*")) if path.is_file()}
        write_json(payload / "manifest.json", files)
        compressed = archive_bytes(payload)
        destination.mkdir(parents=True, exist_ok=True)
        archive = destination / "nvidia_payload.tar.gz"
        archive.write_bytes(compressed)
        embedded = lzma.compress(gzip.decompress(compressed), preset=9)
        notebook_path = destination / "nvidia_colab.ipynb"
        write_json(notebook_path, notebook(embedded))
        write_json(destination / "package.json", {
            "archive_sha256": digest(compressed), "notebook_sha256": digest(notebook_path.read_bytes()),
            "embedded_payload_sha256": digest(embedded), "embedded_compression": "xz",
            "measured_config_sha256": digest(config_bytes),
            "payload_config_sha256": digest((payload / "config.json").read_bytes()),
            "validation_state": "prepared; NVIDIA execution pending",
        })
        print(notebook_path)
        print(f"Embedded archive SHA-256: {digest(embedded)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True,
                        help="Apple session or campaign directory containing the measured inputs")
    parser.add_argument("--output", type=Path, required=True,
                        help="directory for the generated notebook and payload")
    arguments = parser.parse_args()
    prepare(arguments.session.resolve(), arguments.output.resolve())
