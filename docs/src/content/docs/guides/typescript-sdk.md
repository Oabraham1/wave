---
title: TypeScript SDK
description: Getting started with the WAVE TypeScript SDK (wave-gpu) for portable GPU programming.
---

The `wave-gpu` npm package provides a TypeScript-first interface for compiling and launching GPU kernels through the WAVE toolchain. This guide covers installation, device detection, buffer management, kernel compilation, dispatch, and error handling.

## Installation

Requires Node.js 18 or later.

```bash
npm install wave-gpu
```

Import the module:

```typescript
import { array, zeros, ones, device, kernel } from "wave-gpu";
```

## Device Detection

`device()` is async and returns a `Promise<DeviceInfo>`:

```typescript
const dev = await device();
console.log(dev.vendor); // e.g. "Apple"
console.log(dev.name);   // e.g. "Apple M2 Max"
```

The promise rejects if no GPU is found.

## Creating Arrays

`WaveArray` is the buffer type used to transfer data between the host and the GPU. Create one with the factory functions:

```typescript
// From existing data
const a = array([1.0, 2.0, 3.0, 4.0], "f32");

// Pre-filled buffers
const z = zeros(1024, "f32");
const o = ones(1024, "u32");
```

Supported `DType` values: `"f16"`, `"f32"`, `"f64"`, `"i32"`, `"u32"`.

`WaveArray` exposes the following members:

| Member | Description |
|---|---|
| `data` | Raw underlying data |
| `dtype` | Element type string |
| `length` | Number of elements |
| `toArray()` | Copy contents to a JavaScript array |
| `toBuffer()` | Copy contents to a Node.js `Buffer` |

## Writing and Compiling Kernels

Call `kernel()` with a source string and an optional language identifier. The default language is `"typescript"`:

```typescript
const src = `
import wave_gpu

@wave_gpu.kernel
def vector_add(a, b, out, n):
    tid = wave_gpu.thread_id()
    if tid < n:
        out[tid] = a[tid] + b[tid]
`;

const vectorAdd = kernel(src, "python");
```

Supported language values: `"python"`, `"rust"`, `"cpp"`, `"typescript"`.

When the language argument is omitted the source is parsed as TypeScript:

```typescript
const tsKernel = kernel(`
  export function add(a: WaveArray, b: WaveArray, out: WaveArray, n: number) {
    const tid = thread_id();
    if (tid < n) {
      out[tid] = a[tid] + b[tid];
    }
  }
`);
```

## Launching Kernels

`CompiledKernel.launch()` is async. It takes the device, buffer and scalar arrays, and optional grid/workgroup dimensions:

```typescript
const n = 1024;
const a   = array(Float32Array.from({ length: n }, (_, i) => i), "f32");
const b   = ones(n, "f32");
const out = zeros(n, "f32");

const dev = await device();

// Launch with explicit grid and workgroup
await vectorAdd.launch(dev, [a, b, out], [n], [n / 256, 1, 1], [256, 1, 1]);

// Grid and workgroup are optional - WAVE can infer defaults
await vectorAdd.launch(dev, [a, b, out], [n]);
```

The grid and workgroup parameters are 3-element arrays representing `[x, y, z]` dimensions.

## Reading Results

After the launch promise resolves, read data back on the host:

```typescript
const result = out.toArray();
console.log(result.slice(0, 4)); // [1, 2, 3, 4]
```

For interop with Node.js APIs you can also get a `Buffer`:

```typescript
const buf = out.toBuffer();
```

## Error Handling

All async operations (`device()`, `launch()`) signal failures through promise rejection. Use `try`/`catch` with `async`/`await` or `.catch()` on the promise:

```typescript
try {
  const dev = await device();
  await vectorAdd.launch(dev, [a, b, out], [n]);
} catch (err) {
  console.error("WAVE error:", err);
}
```

Synchronous errors (e.g. passing the wrong number of arguments to `kernel()`) throw immediately.

## Next Steps

See the full [TypeScript API Reference](/reference/typescript-api) for detailed type definitions, interfaces, and advanced configuration.
