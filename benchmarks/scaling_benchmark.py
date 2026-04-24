# Copyright 2026 Ojima Abraham
# SPDX-License-Identifier: Apache-2.0

"""Multi-GPU scaling benchmark for WAVE distributed training.

Measures throughput (samples/sec) as the number of simulated devices
increases from 1 to N. On single-GPU systems this benchmarks the
framework overhead of sharding, replication, and all-reduce without
real multi-device communication. Real multi-GPU scaling tests require
NVIDIA multi-T4 or AMD multi-MI300X hardware.

Usage:
    python benchmarks/scaling_benchmark.py [--max-devices 8] [--batch-size 1024]
"""

import argparse
import math
import time
from typing import Any, Dict, List

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / ".." / "sdk" / "python" / "src"))

from wave_gpu.array import WaveArray
from wave_gpu.device import DeviceInfo
from wave_gpu.distributed import (
    allreduce_average,
    gather_shards,
    replicate,
    shard_tensor,
)


def make_devices(n: int) -> List[DeviceInfo]:
    """Create n synthetic device descriptors for benchmarking."""
    return [DeviceInfo("emulator", f"Bench Device {i}") for i in range(n)]


def fake_forward(shard: WaveArray, params: Dict[str, WaveArray]) -> WaveArray:
    """Simulate a forward pass: multiply each element by a weight."""
    weight = params.get("w")
    if weight is None:
        return shard
    out = [s * weight.data[0] for s in shard.data]
    return WaveArray(out, dtype=shard.dtype)


def fake_backward(shard: WaveArray) -> WaveArray:
    """Simulate a backward pass: gradient is 2x the forward output."""
    return WaveArray([v * 2.0 for v in shard.data], dtype=shard.dtype)


def run_benchmark(
    n_devices: int,
    batch_size: int,
    n_iters: int,
) -> float:
    """Run one scaling point and return samples/sec."""
    batch = WaveArray(list(range(batch_size)), dtype="f32")
    weight = WaveArray([0.5], dtype="f32")

    start = time.perf_counter()

    for _ in range(n_iters):
        shards = shard_tensor(batch, n_devices, dim=0)
        weight_replicas = replicate(weight, n_devices)

        outputs: List[WaveArray] = []
        grad_buffers: List[WaveArray] = []

        for i, shard in enumerate(shards):
            params = {"w": weight_replicas[i]}
            out = fake_forward(shard, params)
            outputs.append(out)
            grad = fake_backward(out)
            grad_buffers.append(grad)

        allreduce_average(grad_buffers)

        _ = gather_shards(outputs)

    elapsed = time.perf_counter() - start
    total_samples = batch_size * n_iters
    return total_samples / elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="WAVE multi-GPU scaling benchmark")
    parser.add_argument("--max-devices", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    print(f"WAVE Distributed Scaling Benchmark")
    print(f"Batch size: {args.batch_size}, Iterations: {args.iters}")
    print(f"{'Devices':>8} {'Samples/sec':>14} {'Speedup':>10}")
    print("-" * 36)

    baseline = None
    for n in [1, 2, 4, 8]:
        if n > args.max_devices:
            break
        throughput = run_benchmark(n, args.batch_size, args.iters)
        if baseline is None:
            baseline = throughput
        speedup = throughput / baseline
        print(f"{n:>8} {throughput:>14.1f} {speedup:>10.2f}x")


if __name__ == "__main__":
    main()
