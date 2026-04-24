// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Multi-device kernel launch, tensor sharding, and gradient all-reduce.
//!
//! Extends the WAVE runtime with data-parallel training primitives. Tensors
//! are split along a chosen dimension (typically the batch axis) and each
//! shard is dispatched to a separate device. After each device computes its
//! local gradients, all-reduce averages them across devices. For single-GPU
//! systems (e.g. Apple M-series) these operations are no-ops that preserve
//! correctness and let the same code path run everywhere.

use crate::device::{DeviceInfo, GpuVendor};
use crate::error::RuntimeError;
use crate::launcher::launch_kernel;
use crate::memory::DeviceBuffer;

/// A tensor shard bound to a specific device.
#[derive(Debug)]
pub struct TensorShard {
    /// The device this shard lives on.
    pub device_id: usize,
    /// The buffer holding this shard's data.
    pub buffer: DeviceBuffer,
    /// Offset into the original tensor (in elements).
    pub offset: usize,
    /// Number of elements in this shard.
    pub count: usize,
}

/// Launch a compiled kernel on a specific device.
///
/// Each device uses its own vendor backend. The `device` parameter selects
/// which GPU to target. On single-GPU systems this is equivalent to the
/// regular `launch_kernel` call.
///
/// # Errors
///
/// Returns `RuntimeError::Launch` if the kernel cannot be executed on the
/// specified device.
pub fn launch_on_device(
    vendor_code: &str,
    wbin: &[u8],
    device: &DeviceInfo,
    buffers: &mut [&mut DeviceBuffer],
    scalars: &[u32],
    grid: [u32; 3],
    workgroup: [u32; 3],
) -> Result<(), RuntimeError> {
    launch_kernel(vendor_code, wbin, device.vendor, buffers, scalars, grid, workgroup)
}

/// Split a buffer along the first (batch) dimension across devices.
///
/// Divides the buffer into `devices.len()` contiguous chunks. The last shard
/// absorbs any remainder when the element count is not evenly divisible.
/// With a single device, returns one shard containing the full buffer.
///
/// # Errors
///
/// Returns `RuntimeError::InvalidArgument` if the device list is empty.
pub fn shard_tensor(
    buffer: &DeviceBuffer,
    devices: &[DeviceInfo],
    _dim: usize,
) -> Result<Vec<TensorShard>, RuntimeError> {
    if devices.is_empty() {
        return Err(RuntimeError::InvalidArgument(
            "shard_tensor requires at least one device".into(),
        ));
    }

    let n = buffer.count;
    let n_devices = devices.len();
    let chunk_size = n / n_devices;
    let elem_bytes = buffer.element_type.size_bytes();

    let mut shards = Vec::with_capacity(n_devices);
    let mut offset: usize = 0;

    for (i, dev) in devices.iter().enumerate() {
        let count = if i == n_devices - 1 {
            n - offset
        } else {
            chunk_size
        };

        let byte_start = offset * elem_bytes;
        let byte_end = (offset + count) * elem_bytes;
        let shard_data = buffer.data[byte_start..byte_end].to_vec();

        shards.push(TensorShard {
            device_id: dev.id,
            buffer: DeviceBuffer {
                data: shard_data,
                count,
                element_type: buffer.element_type,
            },
            offset,
            count,
        });

        offset += count;
    }

    Ok(shards)
}

/// Reassemble shards into a single buffer.
///
/// The shards must be in order by offset. The resulting buffer has the
/// combined element count and the element type of the first shard.
///
/// # Errors
///
/// Returns `RuntimeError::InvalidArgument` if the shard list is empty.
pub fn gather_shards(shards: &[TensorShard]) -> Result<DeviceBuffer, RuntimeError> {
    if shards.is_empty() {
        return Err(RuntimeError::InvalidArgument(
            "gather_shards requires at least one shard".into(),
        ));
    }

    let total_count: usize = shards.iter().map(|s| s.count).sum();
    let mut data = Vec::with_capacity(total_count * shards[0].buffer.element_type.size_bytes());

    for shard in shards {
        data.extend_from_slice(&shard.buffer.data);
    }

    Ok(DeviceBuffer {
        data,
        count: total_count,
        element_type: shards[0].buffer.element_type,
    })
}

/// Reduce strategy selected based on device topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceStrategy {
    /// Single device, no communication needed.
    SingleDevice,
    /// All devices share the same vendor, use vendor-native collectives.
    SameVendor(GpuVendor),
    /// Mixed vendors, fall back to host-side reduction.
    CrossVendor,
}

/// Determine the best reduce strategy for a set of devices.
pub fn select_reduce_strategy(devices: &[DeviceInfo]) -> ReduceStrategy {
    if devices.len() <= 1 {
        return ReduceStrategy::SingleDevice;
    }

    let first_vendor = devices[0].vendor;
    if devices.iter().all(|d| d.vendor == first_vendor) {
        ReduceStrategy::SameVendor(first_vendor)
    } else {
        ReduceStrategy::CrossVendor
    }
}

/// All-reduce: average gradient buffers across devices in place.
///
/// After each device computes local gradients, this function averages them
/// so every device holds the same mean gradient. The strategy is chosen
/// automatically:
///
/// - Single device: no-op (returns input unchanged).
/// - Same vendor: host-side averaging (vendor-native NCCL/RCCL planned).
/// - Cross-vendor: host-side averaging via CPU.
///
/// All buffers must have the same element count and type (`f32`).
///
/// # Errors
///
/// Returns `RuntimeError::InvalidArgument` if buffers are empty or have
/// mismatched sizes. Returns `RuntimeError::Memory` on type mismatch.
pub fn allreduce_average(buffers: &mut [DeviceBuffer]) -> Result<(), RuntimeError> {
    if buffers.is_empty() {
        return Err(RuntimeError::InvalidArgument(
            "allreduce_average requires at least one buffer".into(),
        ));
    }

    if buffers.len() == 1 {
        return Ok(());
    }

    let count = buffers[0].count;
    for buf in buffers.iter() {
        if buf.count != count {
            return Err(RuntimeError::InvalidArgument(format!(
                "allreduce buffer size mismatch: expected {count}, got {}",
                buf.count
            )));
        }
    }

    let n_buffers = buffers.len();
    let values: Vec<Vec<f32>> = buffers
        .iter()
        .map(DeviceBuffer::to_f32)
        .collect::<Result<Vec<_>, _>>()?;

    let mut averaged = vec![0.0_f32; count];
    for vals in &values {
        for (i, &v) in vals.iter().enumerate() {
            averaged[i] += v;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let scale = 1.0 / n_buffers as f32;
    for val in &mut averaged {
        *val *= scale;
    }

    let result = DeviceBuffer::from_f32(&averaged);
    for buf in buffers.iter_mut() {
        buf.data.clone_from(&result.data);
    }

    Ok(())
}

/// Replicate a buffer to all devices (for broadcasting model weights).
///
/// Returns one clone of the buffer per device. On a single device this
/// returns a one-element vector containing a clone of the input.
pub fn replicate_buffer(buffer: &DeviceBuffer, devices: &[DeviceInfo]) -> Vec<DeviceBuffer> {
    devices.iter().map(|_| buffer.clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{DeviceCapabilities, DeviceInfo, GpuVendor};
    use crate::memory::DeviceBuffer;

    fn make_device(id: usize, vendor: GpuVendor) -> DeviceInfo {
        DeviceInfo {
            id,
            vendor,
            name: format!("Test Device {id}"),
            memory_bytes: 1024 * 1024 * 1024,
            wave_width: 32,
            max_registers: 32,
            scratchpad_bytes: 32768,
            capabilities: DeviceCapabilities {
                f16: true,
                bf16: false,
                mma: false,
            },
        }
    }

    #[test]
    fn test_shard_single_device() {
        let buf = DeviceBuffer::from_f32(&[1.0, 2.0, 3.0, 4.0]);
        let devices = vec![make_device(0, GpuVendor::Apple)];
        let shards = shard_tensor(&buf, &devices, 0).unwrap();

        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].count, 4);
        assert_eq!(shards[0].offset, 0);
        assert_eq!(shards[0].buffer.to_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_shard_two_devices() {
        let buf = DeviceBuffer::from_f32(&[1.0, 2.0, 3.0, 4.0]);
        let devices = vec![
            make_device(0, GpuVendor::Nvidia),
            make_device(1, GpuVendor::Nvidia),
        ];
        let shards = shard_tensor(&buf, &devices, 0).unwrap();

        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].buffer.to_f32().unwrap(), vec![1.0, 2.0]);
        assert_eq!(shards[1].buffer.to_f32().unwrap(), vec![3.0, 4.0]);
    }

    #[test]
    fn test_shard_remainder() {
        let buf = DeviceBuffer::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let devices = vec![
            make_device(0, GpuVendor::Nvidia),
            make_device(1, GpuVendor::Nvidia),
        ];
        let shards = shard_tensor(&buf, &devices, 0).unwrap();

        assert_eq!(shards[0].count, 2);
        assert_eq!(shards[1].count, 3);
    }

    #[test]
    fn test_gather_roundtrip() {
        let buf = DeviceBuffer::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let devices = vec![
            make_device(0, GpuVendor::Nvidia),
            make_device(1, GpuVendor::Nvidia),
            make_device(2, GpuVendor::Nvidia),
        ];
        let shards = shard_tensor(&buf, &devices, 0).unwrap();
        let gathered = gather_shards(&shards).unwrap();

        assert_eq!(gathered.to_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_allreduce_single_buffer() {
        let mut buf = DeviceBuffer::from_f32(&[2.0, 4.0]);
        allreduce_average(std::slice::from_mut(&mut buf)).unwrap();
        assert_eq!(buf.to_f32().unwrap(), vec![2.0, 4.0]);
    }

    #[test]
    fn test_allreduce_two_buffers() {
        let buf0 = DeviceBuffer::from_f32(&[2.0, 4.0, 6.0]);
        let buf1 = DeviceBuffer::from_f32(&[4.0, 6.0, 8.0]);

        let mut bufs = vec![buf0, buf1];
        allreduce_average(&mut bufs).unwrap();

        assert_eq!(bufs[0].to_f32().unwrap(), vec![3.0, 5.0, 7.0]);
        assert_eq!(bufs[1].to_f32().unwrap(), vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_reduce_strategy_single() {
        let devices = vec![make_device(0, GpuVendor::Apple)];
        assert_eq!(select_reduce_strategy(&devices), ReduceStrategy::SingleDevice);
    }

    #[test]
    fn test_reduce_strategy_same_vendor() {
        let devices = vec![
            make_device(0, GpuVendor::Nvidia),
            make_device(1, GpuVendor::Nvidia),
        ];
        assert_eq!(
            select_reduce_strategy(&devices),
            ReduceStrategy::SameVendor(GpuVendor::Nvidia)
        );
    }

    #[test]
    fn test_reduce_strategy_cross_vendor() {
        let devices = vec![
            make_device(0, GpuVendor::Nvidia),
            make_device(1, GpuVendor::Amd),
        ];
        assert_eq!(select_reduce_strategy(&devices), ReduceStrategy::CrossVendor);
    }

    #[test]
    fn test_replicate_buffer() {
        let buf = DeviceBuffer::from_f32(&[1.0, 2.0]);
        let devices = vec![
            make_device(0, GpuVendor::Nvidia),
            make_device(1, GpuVendor::Nvidia),
        ];
        let replicas = replicate_buffer(&buf, &devices);
        assert_eq!(replicas.len(), 2);
        assert_eq!(replicas[0].to_f32().unwrap(), vec![1.0, 2.0]);
        assert_eq!(replicas[1].to_f32().unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_shard_empty_devices() {
        let buf = DeviceBuffer::from_f32(&[1.0]);
        let result = shard_tensor(&buf, &[], 0);
        assert!(result.is_err());
    }
}
