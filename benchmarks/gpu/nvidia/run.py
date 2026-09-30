"""Validate frozen WAVE binaries and measure paired CUDA dispatches."""

import argparse
import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import struct
import subprocess
import sys
import time
import traceback

import rounding


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_payload(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"invalid manifest path: {name}")
        if digest(path.read_bytes()) != expected:
            raise ValueError(f"payload checksum mismatch: {name}")
    return manifest


def expected_for_dispatches(expected, case, dispatches):
    if not case["name"].startswith("histogram_"):
        return expected
    result = bytearray(expected)
    first = case["output_word_offset"]
    for index in range(first, first + case["output_words"]):
        value, = struct.unpack_from("<I", expected, index * 4)
        if value * dispatches > 0xFFFFFFFF:
            raise ValueError("histogram accumulation exceeds u32")
        struct.pack_into("<I", result, index * 4, value * dispatches)
    return bytes(result)


def check_memory(actual, expected, context):
    if actual == expected:
        return digest(actual)
    if len(actual) != len(expected):
        raise ValueError(f"{context}: memory length differs")
    for byte in range(0, len(actual), 4):
        if actual[byte:byte + 4] != expected[byte:byte + 4]:
            got, = struct.unpack_from("<I", actual, byte)
            want, = struct.unpack_from("<I", expected, byte)
            raise ValueError(f"{context}: word {byte // 4}: {got:#010x}, expected {want:#010x}")
    raise ValueError(f"{context}: memory differs")


class Driver:
    """Bind the CUDA Driver API exported by the installed NVIDIA driver."""

    def __init__(self):
        self.library = ct.CDLL("libcuda.so.1")
        pointer = ct.c_void_p
        signatures = {
            "cuInit": [ct.c_uint],
            "cuDeviceGet": [ct.POINTER(ct.c_int), ct.c_int],
            "cuDeviceGetName": [pointer, ct.c_int, ct.c_int],
            "cuDeviceComputeCapability": [ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), ct.c_int],
            "cuDriverGetVersion": [ct.POINTER(ct.c_int)],
            "cuCtxCreate_v2": [ct.POINTER(pointer), ct.c_uint, ct.c_int],
            "cuCtxDestroy_v2": [pointer],
            "cuCtxSynchronize": [],
            "cuMemAlloc_v2": [ct.POINTER(ct.c_uint64), ct.c_size_t],
            "cuMemFree_v2": [ct.c_uint64],
            "cuMemcpyHtoD_v2": [ct.c_uint64, pointer, ct.c_size_t],
            "cuMemcpyDtoH_v2": [pointer, ct.c_uint64, ct.c_size_t],
            "cuModuleLoad": [ct.POINTER(pointer), ct.c_char_p],
            "cuModuleUnload": [pointer],
            "cuModuleGetFunction": [ct.POINTER(pointer), pointer, ct.c_char_p],
            "cuLaunchKernel": [pointer] + [ct.c_uint] * 7 + [pointer, pointer, pointer],
            "cuEventCreate": [ct.POINTER(pointer), ct.c_uint],
            "cuEventRecord": [pointer, pointer],
            "cuEventSynchronize": [pointer],
            "cuEventElapsedTime": [ct.POINTER(ct.c_float), pointer, pointer],
            "cuEventDestroy_v2": [pointer],
            "cuGetErrorName": [ct.c_int, ct.POINTER(ct.c_char_p)],
            "cuGetErrorString": [ct.c_int, ct.POINTER(ct.c_char_p)],
        }
        for name, parameters in signatures.items():
            function = getattr(self.library, name)
            function.argtypes = parameters
            function.restype = ct.c_int
        self.call("cuInit", 0)
        self.device = ct.c_int()
        self.call("cuDeviceGet", ct.byref(self.device), 0)
        name = ct.create_string_buffer(256)
        major, minor, version = ct.c_int(), ct.c_int(), ct.c_int()
        self.call("cuDeviceGetName", name, len(name), self.device)
        self.call("cuDeviceComputeCapability", ct.byref(major), ct.byref(minor), self.device)
        self.call("cuDriverGetVersion", ct.byref(version))
        self.info = {"name": name.value.decode(), "compute_capability": [major.value, minor.value],
                     "driver_api_version": version.value}
        if (major.value, minor.value) < (7, 5):
            raise ValueError("the frozen PTX requires compute capability 7.5 or newer")
        self.context = ct.c_void_p()
        self.call("cuCtxCreate_v2", ct.byref(self.context), 0, self.device)
        self.modules = []

    def call(self, name, *arguments):
        result = getattr(self.library, name)(*arguments)
        if result:
            error_name, message = ct.c_char_p(), ct.c_char_p()
            self.library.cuGetErrorName(result, ct.byref(error_name))
            self.library.cuGetErrorString(result, ct.byref(message))
            raise RuntimeError(f"{name}: {error_name.value!r}: {message.value!r}")

    def load(self, path, names):
        module = ct.c_void_p()
        self.call("cuModuleLoad", ct.byref(module), str(path).encode())
        self.modules.append(module)
        result = {}
        for name in names:
            function = ct.c_void_p()
            self.call("cuModuleGetFunction", ct.byref(function), module, name.encode())
            result[name] = function
        return result

    def close(self):
        try:
            for module in reversed(self.modules):
                self.call("cuModuleUnload", module)
        finally:
            self.call("cuCtxDestroy_v2", self.context)


class Allocation:
    """Keep one case resident while checking the complete memory allocation."""

    def __init__(self, driver, initial):
        self.driver = driver
        self.initial = ct.create_string_buffer(initial, len(initial))
        self.host = ct.create_string_buffer(len(initial))
        self.size = len(initial)
        self.device = ct.c_uint64()
        driver.call("cuMemAlloc_v2", ct.byref(self.device), self.size)
        self.parameters = (ct.c_void_p * 1)(ct.addressof(self.device))
        self.start, self.end = ct.c_void_p(), ct.c_void_p()
        driver.call("cuEventCreate", ct.byref(self.start), 0)
        driver.call("cuEventCreate", ct.byref(self.end), 0)

    def execute(self, function, case, dispatches):
        call = self.driver.call
        call("cuMemcpyHtoD_v2", self.device, self.initial, self.size)
        call("cuCtxSynchronize")
        host_start = time.perf_counter_ns()
        call("cuEventRecord", self.start, None)
        for _ in range(dispatches):
            call("cuLaunchKernel", function, case["groups"], 1, 1,
                 case["threads"], 1, 1, 0, None, self.parameters, None)
        call("cuEventRecord", self.end, None)
        call("cuEventSynchronize", self.end)
        host_ns = time.perf_counter_ns() - host_start
        elapsed = ct.c_float()
        call("cuEventElapsedTime", ct.byref(elapsed), self.start, self.end)
        if not math.isfinite(elapsed.value) or elapsed.value < 0:
            raise ValueError(f"invalid CUDA event duration: {elapsed.value}")
        call("cuMemcpyDtoH_v2", self.host, self.device, self.size)
        return self.host.raw, {"event_ms": elapsed.value,
                               "event_us_per_dispatch": elapsed.value * 1000 / dispatches,
                               "host_ns": host_ns}

    def close(self):
        for event in (self.start, self.end):
            self.driver.call("cuEventDestroy_v2", event)
        self.driver.call("cuMemFree_v2", self.device)


def command(arguments, log_path):
    start = time.perf_counter()
    result = subprocess.run(arguments, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(result.stdout)
    record = {"argv": arguments, "returncode": result.returncode,
              "wall_seconds": time.perf_counter() - start, "log": log_path.name}
    if result.returncode:
        raise RuntimeError(f"command failed: {arguments!r}\n{result.stdout}")
    return record


def compile_kernels(root, output, driver, names):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is unavailable; select a GPU runtime with the CUDA toolkit")
    ptxas = shutil.which("ptxas") or str(Path(nvcc).resolve().with_name("ptxas"))
    nvdisasm = shutil.which("nvdisasm") or str(Path(nvcc).resolve().with_name("nvdisasm"))
    if not Path(nvdisasm).is_file():
        raise RuntimeError("nvdisasm is required to retain machine code for every compiled kernel")
    major, minor = driver.info["compute_capability"]
    architecture = f"sm_{major}{minor}"
    records, disassembly = [], []

    def execute(arguments, log):
        records.append(command(arguments, output / log))
        write_json(output / "compilation.json", records)

    def inspect(cubin, kernels, variant):
        sass = cubin.with_suffix(".sass")
        execute([nvdisasm, str(cubin)], sass.name)
        text = sass.read_text()
        if not text.strip() or any(name not in text for name in kernels):
            raise ValueError(f"disassembly is empty or lacks a kernel: {sass.name}")
        disassembly.append({"variant": variant, "kernels": kernels,
                            "cubin": cubin.name, "cubin_sha256": digest(cubin.read_bytes()),
                            "sass": sass.name, "sass_sha256": digest(sass.read_bytes())})
        write_json(output / "disassembly.json", disassembly)

    execute([nvcc, "--version"], "nvcc-version.txt")
    execute([ptxas, "--version"], "ptxas-version.txt")
    execute([nvdisasm, "--version"], "nvdisasm-version.txt")
    execute([nvcc, "--ptx", "--gpu-architecture=compute_75", "-O3", "--fmad=false",
             str(root / "native.cu"), "-o", str(output / "native.ptx")], "native-nvcc.txt")
    execute([ptxas, "--gpu-name", architecture, "-O3", "-v",
             str(output / "native.ptx"), "-o", str(output / "native.cubin")], "native-ptxas.txt")
    inspect(output / "native.cubin", names, "native")
    native = driver.load(output / "native.cubin", names)
    wave = {}
    for name in names:
        cubin = output / f"{name}.wave.cubin"
        execute([ptxas, "--gpu-name", architecture, "-O3", "-v",
                 str(root / "data" / f"{name}.wave.ptx"), "-o", str(cubin)], f"{name}-ptxas.txt")
        inspect(cubin, [name], "wave")
        wave[name] = driver.load(cubin, [name])[name]
    witness = {}
    name = rounding.NAME
    execute([nvcc, "--ptx", "--gpu-architecture=compute_75", "-O3", "--fmad=false",
             str(root / "rounding.cu"), "-o", str(output / f"{name}.native.ptx")],
            f"{name}-nvcc.txt")
    for variant in ("wave", "native"):
        source = root / "data" / f"{name}.wave.ptx" if variant == "wave" else output / f"{name}.native.ptx"
        cubin = output / f"{name}.{variant}.cubin"
        execute([ptxas, "--gpu-name", architecture, "-O3", "-v", str(source), "-o", str(cubin)],
                f"{name}-{variant}-ptxas.txt")
        inspect(cubin, [name], variant)
        witness[variant] = driver.load(cubin, [name])[name]
    return {"wave": wave, "native": native, "rounding": witness}


def rounding_payload(root):
    spec = json.loads((root / "rounding-check.json").read_text())
    rounding.validate_contract(spec)
    initial = (root / "data" / spec["initial_path"]).read_bytes()
    expected = (root / "data" / spec["expected_path"]).read_bytes()
    if initial != rounding.initial_bytes() or expected != rounding.expected_bytes():
        raise ValueError("rounding witness literal memory differs")
    for field, data in (("initial", initial), ("expected", expected),
                        ("wbin", (root / "data" / spec["wbin_path"]).read_bytes()),
                        ("ptx", (root / "data" / spec["ptx_path"]).read_bytes())):
        if spec[field + "_sha256"] != digest(data):
            raise ValueError(f"rounding witness {field} checksum differs")
    if spec["emulator"] != {"correct": True, "dispatches": 1, "memory_sha256": digest(expected)}:
        raise ValueError("rounding witness emulator result differs")
    return spec, initial, expected


def check_rounding(root, output, driver, functions):
    spec, initial, expected = rounding_payload(root)
    checks = []
    allocation = Allocation(driver, initial)
    try:
        for variant in ("wave", "native"):
            actual, _ = allocation.execute(functions[variant], spec, 1)
            actual_path = output / f"rounding-{variant}.actual.bin"
            actual_path.write_bytes(actual)
            checksum = check_memory(actual, expected, f"rounding/{variant}")
            checks.append({"variant": variant, "correct": True, "dispatches": 1,
                           "memory_sha256": checksum, "actual_path": actual_path.name,
                           "output_words": list(struct.unpack("<8I", actual))[4:7]})
    finally:
        allocation.close()
    write_json(output / "rounding-check.json", {
        "state": "completed", "timed": False, "checks": checks,
        "payload_spec_sha256": digest((root / "rounding-check.json").read_bytes()),
        "initial_sha256": digest(initial), "expected_sha256": digest(expected),
        "wbin_sha256": spec["wbin_sha256"], "ptx_sha256": spec["ptx_sha256"],
    })
    print("Literal f32 rounding witness matches on WAVE and native CUDA.", flush=True)


def verify_rounding_record(root, output):
    spec, initial, expected = rounding_payload(root)
    record = json.loads((output / "rounding-check.json").read_text())
    checks = []
    for variant in ("wave", "native"):
        name = f"rounding-{variant}.actual.bin"
        actual = (output / name).read_bytes()
        checksum = check_memory(actual, expected, f"retained rounding/{variant}")
        checks.append({"variant": variant, "correct": True, "dispatches": 1,
                       "memory_sha256": checksum, "actual_path": name,
                       "output_words": list(struct.unpack("<8I", actual))[4:7]})
    expected_record = {"state": "completed", "timed": False, "checks": checks,
                       "payload_spec_sha256": digest((root / "rounding-check.json").read_bytes()),
                       "initial_sha256": digest(initial), "expected_sha256": digest(expected),
                       "wbin_sha256": spec["wbin_sha256"], "ptx_sha256": spec["ptx_sha256"]}
    if record != expected_record:
        raise ValueError("retained rounding witness record differs")


def verify_disassembly(output, names):
    records = json.loads((output / "disassembly.json").read_text())
    expected = {("native", tuple(names), "native.cubin", "native.sass")}
    expected.update(("wave", (name,), f"{name}.wave.cubin", f"{name}.wave.sass") for name in names)
    expected.update((variant, (rounding.NAME,), f"{rounding.NAME}.{variant}.cubin",
                     f"{rounding.NAME}.{variant}.sass") for variant in ("wave", "native"))
    seen = set()
    for record in records:
        key = (record["variant"], tuple(record["kernels"]), record["cubin"], record["sass"])
        if key not in expected or key in seen:
            raise ValueError("disassembly coverage differs")
        seen.add(key)
        for field in ("cubin", "sass"):
            if digest((output / record[field]).read_bytes()) != record[field + "_sha256"]:
                raise ValueError(f"disassembly artifact checksum differs: {record[field]}")
        text = (output / record["sass"]).read_text()
        if not text.strip() or any(name not in text for name in record["kernels"]):
            raise ValueError("disassembly is empty or lacks a kernel")
    if seen != expected:
        raise ValueError("disassembly coverage is incomplete")
    commands = json.loads((output / "compilation.json").read_text())
    for record in records:
        matches = [command for command in commands if command["log"] == record["sass"]]
        if len(matches) != 1 or matches[0]["returncode"] != 0:
            raise ValueError("disassembly command is missing or failed")


def summarize(rows, seed):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["name"], row["size"]), {}).setdefault(row["pair"], {})[row["variant"]] = row
    summary = []
    rng = random.Random(seed)
    for (name, size), pairs in sorted(grouped.items()):
        wave = [pair["wave"]["event_us_per_dispatch"] for pair in pairs.values()]
        native = [pair["native"]["event_us_per_dispatch"] for pair in pairs.values()]
        item = {"name": name, "size": size, "wave_median_us": statistics.median(wave),
                "native_median_us": statistics.median(native), "pairs": len(pairs)}
        if all(value > 0 for value in wave + native):
            logs = [math.log(w / n) for w, n in zip(wave, native)]
            bootstrap = sorted(math.exp(statistics.mean(rng.choices(logs, k=len(logs))))
                               for _ in range(2000))
            item.update({"wave_over_native_geomean": math.exp(statistics.mean(logs)),
                         "wave_over_native_bootstrap_95": [bootstrap[49], bootstrap[1949]]})
        else:
            item["ratio_unavailable"] = "event duration reached timer resolution"
        summary.append(item)
    return summary


def run(root, output, repeats, warmup, dispatches, seed):
    manifest = verify_payload(root)
    config = json.loads((root / "config.json").read_text())
    cases = list(config["cases"])
    identities = {(case["name"], case["size"]) for case in cases}
    if not cases or len(identities) != len(cases):
        raise ValueError("payload cases are missing or duplicated")
    rng = random.Random(seed)
    rng.shuffle(cases)
    write_json(output / "protocol.json", {
        "repeats": repeats, "warmup": warmup, "dispatches_per_sample": dispatches,
        "order_seed": seed, "case_order": [[c["name"], c["size"]] for c in cases],
        "correctness": "bit-exact comparison of the full allocation after every sample",
        "event_timing": "CUDA events enclose kernel submissions on the default stream; host submission gaps may contribute",
        "host_timing": "event recording, kernel submission and event synchronization; transfers excluded",
        "excluded": ["translation", "native compilation", "ptxas", "module loading", "allocation", "transfers", "validation"],
        "statistics": "geometric mean of paired time ratios; percentile bootstrap of pairs within this session",
        "payload_manifest_sha256": digest((root / "manifest.json").read_bytes()),
    })
    driver = Driver()
    environment = {"device": driver.info, "platform": platform.platform(), "python": sys.version,
                   "process_id": os.getpid()}
    write_json(output / "environment.json", environment)
    try:
        if shutil.which("nvidia-smi"):
            command(["nvidia-smi", "-q"], output / "nvidia-smi-before.txt")
        functions = compile_kernels(root, output, driver, sorted({c["name"] for c in cases}))
        verify_disassembly(output, sorted({c["name"] for c in cases}))
        check_rounding(root, output, driver, functions["rounding"])
        verify_rounding_record(root, output)
        rows, checks = [], []
        with (output / "samples.jsonl").open("w") as samples:
            for case in cases:
                name, size = case["name"], case["size"]
                initial = (root / "data" / case["initial_path"]).read_bytes()
                expected = (root / "data" / case["expected_path"]).read_bytes()
                repeated = expected_for_dispatches(expected, case, dispatches)
                allocation = Allocation(driver, initial)
                try:
                    for variant in ("wave", "native"):
                        actual, _ = allocation.execute(functions[variant][name], case, 1)
                        checksum = check_memory(actual, expected, f"{name}/{size}/{variant}/single")
                        checks.append({"name": name, "size": size, "variant": variant,
                                       "dispatches": 1, "memory_sha256": checksum, "correct": True})
                        write_json(output / "correctness.json", checks)
                    for pair in range(-warmup, repeats):
                        order = ["wave", "native"]
                        rng.shuffle(order)
                        for position, variant in enumerate(order):
                            actual, timing = allocation.execute(functions[variant][name], case, dispatches)
                            checksum = check_memory(actual, repeated, f"{name}/{size}/{variant}/{pair}")
                            row = {"name": name, "size": size, "variant": variant, "pair": pair,
                                   "position": position, "dispatches": dispatches, "correct": True,
                                   "memory_sha256": checksum, "warmup": pair < 0, **timing}
                            samples.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                            samples.flush()
                            if pair >= 0:
                                rows.append(row)
                    print(f"{name} size={size}: complete memory matches", flush=True)
                finally:
                    allocation.close()
        write_json(output / "summary.json", summarize(rows, seed))
        if shutil.which("nvidia-smi"):
            command(["nvidia-smi", "-q"], output / "nvidia-smi-after.txt")
    finally:
        driver.close()
    write_json(output / "payload-manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dispatches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_payload(args.payload)
        print("Payload checksums match.")
        return
    if args.repeats <= 0 or args.warmup < 0 or args.dispatches <= 0:
        parser.error("repeats and dispatches must be positive; warmup must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    status = {"state": "running", "started_unix_seconds": time.time()}
    write_json(args.output / "status.json", status)
    try:
        run(args.payload.resolve(), args.output.resolve(), args.repeats, args.warmup,
            args.dispatches, args.seed)
        status["state"] = "completed"
    except Exception as error:
        status.update({"state": "failed", "error": str(error)})
        (args.output / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        status["finished_unix_seconds"] = time.time()
        write_json(args.output / "status.json", status)
        hashes = {str(path.relative_to(args.output)): digest(path.read_bytes())
                  for path in sorted(args.output.rglob("*")) if path.is_file()}
        write_json(args.output / "artifacts.json", hashes)


if __name__ == "__main__":
    main()
