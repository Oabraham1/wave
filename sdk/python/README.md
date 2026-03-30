# WAVE Python SDK

Write GPU kernels in Python, run on any GPU.

## Install

```bash
pip install wave-gpu
```

## Usage

```python
import wave_gpu

@wave_gpu.kernel
def vector_add(a: wave_gpu.f32[:], b: wave_gpu.f32[:], out: wave_gpu.f32[:], n: wave_gpu.u32):
    gid = wave_gpu.thread_id()
    if gid < n:
        out[gid] = a[gid] + b[gid]

a = wave_gpu.array([1.0, 2.0, 3.0, 4.0])
b = wave_gpu.array([5.0, 6.0, 7.0, 8.0])
out = wave_gpu.zeros(4)
vector_add(a, b, out, len(a))
print(out.to_list())  # [6.0, 8.0, 10.0, 12.0]
```

## License

Apache 2.0 - see [LICENSE](../../LICENSE)
