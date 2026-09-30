// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Split tensors across devices, launch each shard, and average gradients on the host.

use crate::device::{DeviceInfo, GpuVendor};
use crate::error::RuntimeError;
use crate::launcher::launch_kernel;
use crate::memory::DeviceBuffer;
use num_traits::ToPrimitive;

/// A tensor shard bound to a specific device.
#[derive(Debug)]
pub struct TensorShard {
    /// The device this shard lives on.
    pub device_id: usize,
    /// The buffer holding this shard's data.
    pub buffer: DeviceBuffer,
    /// Offset into the original tensor, in elements.
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
    launch_kernel(
        vendor_code,
        wbin,
        device.vendor,
        buffers,
        scalars,
        grid,
        workgroup,
    )
}

/// Split a buffer along the first, batch, dimension across devices.
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
#[must_use]
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
/// The host sums values in buffer order, then divides by the number of buffers.
/// Every buffer receives the mean. A single buffer is unchanged.
/// All buffers must have the same element count and type, `f32`.
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

    let count = buffers[0].count;
    for buf in buffers.iter() {
        let _ = buf.iter_f32()?;
        if buf.count != count {
            return Err(RuntimeError::InvalidArgument(format!(
                "allreduce buffer size mismatch: expected {count}, got {}",
                buf.count
            )));
        }
    }

    if buffers.len() == 1 {
        return Ok(());
    }
    let n_buffers = buffers.len();
    let mut averaged = vec![0.0_f32; count];
    for buffer in buffers.iter() {
        for (sum, value) in averaged.iter_mut().zip(buffer.iter_f32()?) {
            *sum += value;
        }
    }
    // The reduction and its normalization use the buffer's f32 precision.
    let scale = 1.0
        / n_buffers.to_f32().ok_or_else(|| {
            RuntimeError::InvalidArgument("buffer count exceeds f32 range".into())
        })?;
    for val in &mut averaged {
        *val *= scale;
    }

    if let Some((first, rest)) = buffers.split_first_mut() {
        for (bytes, value) in first.data.as_chunks_mut::<4>().0.iter_mut().zip(averaged) {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        for buffer in rest {
            buffer.data.copy_from_slice(&first.data);
        }
    }

    Ok(())
}

/// Replicate a buffer to all devices, for broadcasting model weights.
///
/// Returns one clone of the buffer per device. On a single device this
/// returns a one-element vector containing a clone of the input.
#[must_use]
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

        assert_eq!(
            gathered.to_f32().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
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

        let mut buffers = vec![buf0, buf1];
        allreduce_average(&mut buffers).unwrap();

        assert_eq!(buffers[0].to_f32().unwrap(), vec![3.0, 5.0, 7.0]);
        assert_eq!(buffers[1].to_f32().unwrap(), vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_reduce_strategy_single() {
        let devices = vec![make_device(0, GpuVendor::Apple)];
        assert_eq!(
            select_reduce_strategy(&devices),
            ReduceStrategy::SingleDevice
        );
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
        assert_eq!(
            select_reduce_strategy(&devices),
            ReduceStrategy::CrossVendor
        );
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
    #[test]
    fn reduction_preserves_buffer_order_and_float_edges() {
        let mut buffers = vec![
            DeviceBuffer::from_f32(&[1.0e20, -0.0, f32::INFINITY, f32::NAN]),
            DeviceBuffer::from_f32(&[1.0, -0.0, f32::NEG_INFINITY, 2.0]),
            DeviceBuffer::from_f32(&[-1.0e20, -0.0, 0.0, 3.0]),
        ];
        allreduce_average(&mut buffers).unwrap();
        let values = buffers[0].to_f32().unwrap();
        assert_eq!(values[0].to_bits(), 0.0_f32.to_bits());
        assert_eq!(values[1].to_bits(), 0.0_f32.to_bits());
        assert!(values[2].is_nan());
        assert!(values[3].is_nan());
        assert!(buffers.iter().all(|buffer| buffer.data == buffers[0].data));
    }

    #[test]
    fn reduction_rejects_invalid_buffers_without_writing() {
        for invalid in [
            DeviceBuffer {
                count: 2,
                data: vec![0; 4],
                element_type: crate::memory::ElementType::F32,
            },
            DeviceBuffer {
                count: 2,
                data: vec![0; 9],
                element_type: crate::memory::ElementType::F32,
            },
            DeviceBuffer::from_u32(&[1, 2]),
        ] {
            let mut buffers = vec![DeviceBuffer::from_f32(&[2.0, 4.0]), invalid];
            let bytes: Vec<_> = buffers.iter().map(|buffer| buffer.data.clone()).collect();
            assert!(allreduce_average(&mut buffers).is_err());
            assert_eq!(
                buffers
                    .iter()
                    .map(|buffer| &buffer.data)
                    .collect::<Vec<_>>(),
                bytes.iter().collect::<Vec<_>>()
            );
        }
    }
}
