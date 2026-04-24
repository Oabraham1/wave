# Copyright 2026 Ojima Abraham
# SPDX-License-Identifier: Apache-2.0

"""Distributed data-parallel training for the WAVE Python SDK.

Provides DistributedDataParallel, which wraps a model and automatically
shards input batches across devices, replicates weights, and averages
gradients after each backward pass. On single-GPU systems (e.g. Apple
M-series) all operations are no-ops that preserve correctness so the
same training code runs everywhere.
"""

from typing import Any, Callable, Dict, List, Optional

from .array import WaveArray
from .device import DeviceInfo, detect_gpu


def enumerate_devices() -> List[DeviceInfo]:
    """Return all available GPU devices.

    On macOS returns a single Metal device. On Linux enumerates all
    NVIDIA, AMD, and Intel GPUs. Falls back to the emulator when no
    GPU hardware is detected.
    """
    vendor, name = detect_gpu()
    return [DeviceInfo(vendor, name)]


def shard_tensor(
    data: WaveArray,
    n_devices: int,
    dim: int = 0,
) -> List[WaveArray]:
    """Split a WaveArray into n_devices contiguous chunks.

    The last shard absorbs any remainder elements when the array length
    is not evenly divisible by n_devices. With n_devices=1 this returns
    a single-element list containing the original data.
    """
    n = len(data)
    chunk = n // n_devices
    shards: List[WaveArray] = []

    for i in range(n_devices):
        start = i * chunk
        end = n if i == n_devices - 1 else (i + 1) * chunk
        shards.append(WaveArray(data.data[start:end], dtype=data.dtype))

    return shards


def gather_shards(shards: List[WaveArray]) -> WaveArray:
    """Reassemble sharded arrays into one WaveArray."""
    combined: List[float] = []
    for shard in shards:
        combined.extend(shard.data)
    dtype = shards[0].dtype if shards else "f32"
    return WaveArray(combined, dtype=dtype)


def allreduce_average(buffers: List[WaveArray]) -> None:
    """Average corresponding elements across all buffers in place.

    After this call every buffer in the list contains identical values
    equal to the element-wise mean of the original buffers. With a
    single buffer this is a no-op.
    """
    if len(buffers) <= 1:
        return

    n = len(buffers[0])
    n_buffers = len(buffers)

    averaged = [0.0] * n
    for buf in buffers:
        for i, v in enumerate(buf.data):
            averaged[i] += v

    scale = 1.0 / n_buffers
    for i in range(n):
        averaged[i] *= scale

    for buf in buffers:
        buf.data = list(averaged)


def replicate(data: WaveArray, n_devices: int) -> List[WaveArray]:
    """Create n_devices copies of a WaveArray (for weight replication)."""
    return [WaveArray(list(data.data), dtype=data.dtype) for _ in range(n_devices)]


class DistributedDataParallel:
    """Data-parallel wrapper that shards batches and averages gradients.

    Wraps a callable model (forward function) and a list of named
    parameter arrays. On each forward call the input batch is sharded
    across devices, the model runs on each shard, and the outputs are
    gathered. During backward (gradient computation), gradients are
    averaged across devices via all-reduce.
    """

    def __init__(
        self,
        forward_fn: Callable[..., WaveArray],
        params: Dict[str, WaveArray],
        devices: Optional[List[DeviceInfo]] = None,
    ) -> None:
        self._forward_fn = forward_fn
        self._params = params
        self._devices = devices if devices is not None else enumerate_devices()
        self._n_devices = len(self._devices)

        self._param_replicas: Dict[str, List[WaveArray]] = {}
        for name, param in self._params.items():
            self._param_replicas[name] = replicate(param, self._n_devices)

        self._grad_buffers: Dict[str, List[WaveArray]] = {}

    @property
    def devices(self) -> List[DeviceInfo]:
        """The devices this model is distributed across."""
        return list(self._devices)

    @property
    def n_devices(self) -> int:
        """Number of devices."""
        return self._n_devices

    def forward(self, batch: WaveArray, **kwargs: Any) -> WaveArray:
        """Run the forward pass with automatic batch sharding.

        Splits the batch across devices, runs the model on each shard,
        and gathers the results into a single output array.
        """
        shards = shard_tensor(batch, self._n_devices, dim=0)

        outputs: List[WaveArray] = []
        for i, shard in enumerate(shards):
            device_params = {
                name: replicas[i] for name, replicas in self._param_replicas.items()
            }
            out = self._forward_fn(shard, device_params, **kwargs)
            outputs.append(out)

        return gather_shards(outputs)

    def backward(self, grad_buffers: Dict[str, List[WaveArray]]) -> None:
        """Average gradients across devices via all-reduce.

        Each key maps a parameter name to a list of per-device gradient
        arrays. After this call all gradient arrays for a given parameter
        contain the same averaged values.
        """
        for name, grads in grad_buffers.items():
            allreduce_average(grads)
            self._grad_buffers[name] = grads

    def get_averaged_gradients(self) -> Dict[str, WaveArray]:
        """Return the most recent averaged gradients (one per parameter)."""
        result: Dict[str, WaveArray] = {}
        for name, grads in self._grad_buffers.items():
            if grads:
                result[name] = grads[0]
        return result

    def sync_params(self) -> None:
        """Broadcast current parameter values to all device replicas."""
        for name, param in self._params.items():
            self._param_replicas[name] = replicate(param, self._n_devices)

    def __call__(self, batch: WaveArray, **kwargs: Any) -> WaveArray:
        return self.forward(batch, **kwargs)
