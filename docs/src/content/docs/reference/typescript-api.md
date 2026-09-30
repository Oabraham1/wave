---
title: "TypeScript API reference"
description: "Arrays, device detection, compilation, and emulator dispatch in wave-gpu."
---

The `wave-gpu` package provides a Node.js API for host arrays and device detection. Its launch wrapper is incompatible with the current emulator CLI. Use the [Python quickstart](/getting-started/quickstart/) or [Rust SDK](/guides/rust-sdk/) for execution.

## Installation

```bash
npm install wave-gpu
```

## Types

### `DType`

```typescript
type DType = "f32" | "u32" | "i32" | "f16" | "f64";
```

### `GpuVendor`

```typescript
type GpuVendor = "apple" | "nvidia" | "amd" | "intel" | "emulator";
```

### `Language`

```typescript
type Language = "python" | "rust" | "cpp" | "typescript";
```

### `DeviceInfo`

```typescript
interface DeviceInfo {
  vendor: GpuVendor;
  name: string;
}
```

### `WaveArray`

A host array with a dtype label.

```typescript
declare class WaveArray {
  data: number[];
  dtype: DType;
  constructor(data: number[], dtype?: DType);
  get length(): number;
  toArray(): number[];
  toBuffer(): Buffer;
  static fromBuffer(buf: Buffer, count: number, dtype?: DType): WaveArray;
}
```

### `CompiledKernel`

Stores source and caches the compiled binary after the first launch.

```typescript
declare class CompiledKernel {
  constructor(source: string, language?: Language);
  launch(
    dev: DeviceInfo,
    buffers: WaveArray[],
    scalars: number[],
    grid?: [number, number, number],
    workgroup?: [number, number, number],
  ): Promise<void>;
}
```

## Functions

### `array(data, dtype?) -> WaveArray`

Copy a JavaScript `number[]` into a host array. The dtype defaults to `"f32"`.

```typescript
import { array } from "wave-gpu";

const a = array([1.0, 2.0, 3.0, 4.0]);
```

### `zeros(n, dtype?) -> WaveArray`

Create a host array of `n` zeros. The dtype defaults to `"f32"`.

### `ones(n, dtype?) -> WaveArray`

Create a host array of `n` ones. The dtype defaults to `"f32"`.

### `device() -> Promise<DeviceInfo>`

Detect a device. Return the emulator if no GPU is found.

```typescript
import { device } from "wave-gpu";

const dev = await device();
console.log(`${dev.vendor}: ${dev.name}`);
```

### `kernel(source, lang?) -> CompiledKernel`

Store kernel source for compilation on first launch. The language defaults to `"typescript"`; it also accepts `"python"`, `"rust"`, and `"cpp"`. Compilation errors reject the launch promise.

```typescript
import { kernel } from "wave-gpu";

const add = kernel(`
  export function add(a: f32[], b: f32[], out: f32[], n: u32): void {
    const tid = thread_id();
    if (tid < n) {
      out[tid] = a[tid] + b[tid];
    }
  }
`);
```

## WaveArray methods

### `toArray() -> number[]`

Return a copy of the stored values.

### `toBuffer() -> Buffer`

Encode values as little-endian `f32` bytes. This method uses four bytes per element regardless of the dtype label. `WaveArray.fromBuffer` decodes the same format.

### `length`: property

The number of stored elements.

### `data`: property

The mutable `number[]` containing host values.

### `dtype`: property

The element type label.

## CompiledKernel.launch: async

The method compiles if needed, then attempts emulator execution. Its wrapper passes `--memory-file`, `--workgroup`, and `--reg` options that the current `wave-emu` CLI does not accept, so the promise rejects. Other device vendors are rejected directly.

This example shows the argument order; it does not complete execution with the current CLI.

```typescript
await add.launch(
  { vendor: "emulator", name: "WAVE Emulator" },
  [a, b, out],
  [4],
  [1, 1, 1],
  [4, 1, 1],
);
```

| Argument | Default | Meaning |
|---|---|---|
| `dev` | Required | Target device |
| `buffers` | Required | Buffer arguments in declaration order |
| `scalars` | Required | Scalar arguments |
| `grid` | `[1, 1, 1]` | Workgroup counts in x, y, and z |
| `workgroup` | `[256, 1, 1]` | Threads per workgroup in x, y, and z |

## Complete example

This example uses host arrays without dispatching a kernel:

```typescript
import { array, zeros } from "wave-gpu";

const a = array([1.0, 2.0, 3.0, 4.0]);
const output = zeros(a.length);
console.log(a.toArray());
console.log(output.toArray());
console.log(a.toBuffer().length); // 16 bytes
```
