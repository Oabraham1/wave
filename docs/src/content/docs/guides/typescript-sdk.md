---
title: "TypeScript SDK"
description: "Host arrays, kernel compilation, and emulator execution in Node.js."
---

The `wave-gpu` package provides host arrays, device detection, and a kernel API for Node.js. Its launch wrapper is incompatible with the current emulator CLI.

## Installation

Requires Node.js 18 or later.

```bash
npm install wave-gpu
```

Import the module:

```typescript
import { array, zeros, ones, device, kernel } from "wave-gpu";
```

## Device detection

`device()` is async and returns a `Promise<DeviceInfo>`:

```typescript
const dev = await device();
console.log(dev.vendor); // e.g. "apple"
console.log(dev.name);   // e.g. "Apple GPU (Metal)"
```

Detection returns the emulator if no GPU is found.

## Creating arrays

`WaveArray` stores host data for kernel arguments. Create arrays with:

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

## Writing and compiling kernels

Call `kernel()` with a source string and an optional language identifier. The default language is `"typescript"`:

```typescript
const src = `
import wave_gpu

@wave_gpu.kernel
def vector_add(a: f32[:], b: f32[:], out: f32[:], n: u32):
    tid = thread_id()
    if tid < n:
        out[tid] = a[tid] + b[tid]
`;

const vectorAdd = kernel(src, "python");
```

Supported language values: `"python"`, `"rust"`, `"cpp"`, `"typescript"`.

When the language argument is omitted the source is parsed as TypeScript:

```typescript
const tsKernel = kernel(`
  export function add(a: f32[], b: f32[], out: f32[], n: u32) {
    const tid = thread_id();
    if (tid < n) {
      out[tid] = a[tid] + b[tid];
    }
  }
`);
```

## Launching kernels

`CompiledKernel.launch()` compiles on first use. Its emulator wrapper passes `--memory-file`, `--workgroup`, and `--reg` options that the current `wave-emu` CLI does not accept, so execution fails. GPU device vendors are rejected directly. Use the [Python quickstart](/getting-started/quickstart/) or [Rust SDK](/guides/rust-sdk/) for execution.

This call illustrates the argument order; it rejects when it reaches emulator execution:

```typescript
const n = 1024;
const a = array(Array.from({ length: n }, (_, i) => i), "f32");
const b   = ones(n, "f32");
const out = zeros(n, "f32");

const dev = { vendor: "emulator" as const, name: "WAVE Emulator" };

// Launch with explicit grid and workgroup
await vectorAdd.launch(dev, [a, b, out], [n], [n / 256, 1, 1], [256, 1, 1]);

// Defaults are one workgroup of 256 threads.
// Specify dimensions when the input needs more threads.
```

The grid and workgroup parameters are 3-element arrays representing `[x, y, z]` dimensions.

## Reading results

Read the values stored in a host array:

```typescript
const values = array([1, 2, 3, 4]);
console.log(values.toArray()); // [1, 2, 3, 4]
```

For interop with Node.js APIs you can also get a `Buffer`:

```typescript
const buf = values.toBuffer();
```

## Error handling

Async operations signal failures through promise rejection. Use `try`/`catch` with `async`/`await` or `.catch()` on the promise:

```typescript
try {
  const dev = await device();
  await vectorAdd.launch(dev, [a, b, out], [n]);
} catch (err) {
  console.error("WAVE error:", err);
}
```

`kernel()` stores the source. Compilation errors reject the first `launch()` call.

## Next steps

See the [TypeScript API reference](/reference/typescript-api) for types and signatures.
