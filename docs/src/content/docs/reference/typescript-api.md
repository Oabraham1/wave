---
title: TypeScript API Reference
description: Complete API reference for the WAVE TypeScript SDK.
---

The WAVE TypeScript SDK provides GPU compute bindings for Node.js and Deno. It exposes array creation, device detection, kernel compilation, and async dispatch.

## Installation

```bash
npm install wave-gpu
```

## Types

### `DType`

```typescript
type DType = "f16" | "f32" | "f64" | "i32" | "u32";
```

### `GpuVendor`

```typescript
type GpuVendor = "AMD" | "NVIDIA" | "Intel" | "Unknown";
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

```typescript
interface WaveArray {
  readonly length: number;
  readonly data: ArrayBuffer;
  readonly dtype: DType;
  toArray(): number[];
  toBuffer(): Buffer;
}
```

### `CompiledKernel`

```typescript
interface CompiledKernel {
  launch(options: {
    device?: DeviceInfo;
    buffers: WaveArray[];
    scalars?: number[];
    grid: [number, number, number];
    workgroup: [number, number, number];
  }): Promise<void>;
}
```

---

## Functions

### `array(data, dtype?) -> WaveArray`

```typescript
function array(data: number[], dtype?: DType): WaveArray;
```

Create a device array from a JavaScript array.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `number[]` | *(required)* | Source data to upload. |
| `dtype` | `DType` | `"f32"` | Element type. |

**Returns:** `WaveArray`

```typescript
import { array } from "wave-gpu";

const a = array([1.0, 2.0, 3.0, 4.0]);
const b = array([1, 2, 3], "u32");
```

---

### `zeros(n, dtype?) -> WaveArray`

```typescript
function zeros(n: number, dtype?: DType): WaveArray;
```

Create a device array of `n` zeros.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `number` | *(required)* | Number of elements. |
| `dtype` | `DType` | `"f32"` | Element type. |

**Returns:** `WaveArray`

```typescript
const buf = zeros(1024);
const ints = zeros(256, "i32");
```

---

### `ones(n, dtype?) -> WaveArray`

```typescript
function ones(n: number, dtype?: DType): WaveArray;
```

Create a device array of `n` ones.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `number` | *(required)* | Number of elements. |
| `dtype` | `DType` | `"f32"` | Element type. |

**Returns:** `WaveArray`

---

### `device() -> DeviceInfo`

```typescript
function device(): DeviceInfo;
```

Detect the first available GPU and return its information.

**Returns:** `DeviceInfo`

**Throws:** `Error` if no supported GPU is found.

```typescript
import { device } from "wave-gpu";

const dev = device();
console.log(`${dev.vendor}: ${dev.name}`);
```

---

### `kernel(source, lang?) -> CompiledKernel`

```typescript
function kernel(source: string, lang?: Language): CompiledKernel;
```

Compile a kernel from source code.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `string` | *(required)* | Kernel source code. |
| `lang` | `Language` | `"typescript"` | Source language. |

**Returns:** `CompiledKernel`

**Throws:** `Error` if compilation fails.

```typescript
import { kernel } from "wave-gpu";

const add = kernel(`
  export function add(a: f32[], b: f32[], out: f32[]): void {
    const tid = threadId();
    out[tid] = a[tid] + b[tid];
  }
`);
```

---

## WaveArray Methods

### `toArray() -> number[]`

Copy the device buffer to the host and return as a JavaScript array.

```typescript
const a = array([1.0, 2.0, 3.0]);
console.log(a.toArray()); // [1, 2, 3]
```

### `toBuffer() -> Buffer`

Copy the device buffer to the host and return as a Node.js `Buffer` of raw bytes.

```typescript
const a = array([1.0, 2.0], "f32");
const buf = a.toBuffer(); // 8 bytes (2 x 4-byte f32)
```

### `length` (property)

The number of elements in the buffer.

```typescript
const a = zeros(512);
console.log(a.length); // 512
```

### `data` (property)

The raw `ArrayBuffer` backing the device data. Accessing this triggers a device-to-host copy.

### `dtype` (property)

The element type string (e.g., `"f32"`, `"u32"`).

---

## CompiledKernel.launch (async)

Dispatch a compiled kernel on the GPU. Returns a `Promise` that resolves when execution completes.

```typescript
await add.launch({
  buffers: [a, b, out],
  grid: [4, 1, 1],
  workgroup: [4, 1, 1],
});
```

**Options:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `device` | `DeviceInfo` | No | Target device. If omitted, uses the default detected device. |
| `buffers` | `WaveArray[]` | Yes | Device buffers bound as kernel arguments, in declaration order. |
| `scalars` | `number[]` | No | Scalar values passed as 32-bit constants. |
| `grid` | `[number, number, number]` | Yes | Global grid dimensions `[x, y, z]`. |
| `workgroup` | `[number, number, number]` | Yes | Workgroup dimensions `[x, y, z]`. |

---

## Complete Example

```typescript
import { array, zeros, device, kernel } from "wave-gpu";

const dev = device();
console.log(`Running on ${dev.name}`);

const a = array([1.0, 2.0, 3.0, 4.0]);
const b = array([5.0, 6.0, 7.0, 8.0]);
const out = zeros(4);

const add = kernel(`
  export function add(a: f32[], b: f32[], out: f32[]): void {
    const tid = threadId();
    out[tid] = a[tid] + b[tid];
  }
`);

await add.launch({
  buffers: [a, b, out],
  grid: [4, 1, 1],
  workgroup: [4, 1, 1],
});

console.log(out.toArray()); // [6, 8, 10, 12]
```
