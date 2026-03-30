# WAVE TypeScript/JavaScript SDK

Write GPU kernels in TypeScript, run on any GPU.

## Install

```bash
npm install wave-gpu
```

## Usage

```typescript
import { device, kernel, array, zeros } from 'wave-gpu';

const dev = await device();
const vectorAdd = kernel(`
    function vector_add(a: f32[], b: f32[], out: f32[], n: u32) {
        const gid = thread_id();
        if (gid < n) { out[gid] = a[gid] + b[gid]; }
    }
`, 'typescript');

const a = array([1, 2, 3, 4]);
const b = array([5, 6, 7, 8]);
const out = zeros(4);
await vectorAdd.launch(dev, [a, b, out], [4], [1,1,1], [256,1,1]);
console.log(out.toArray());
```

## License

Apache 2.0 - see [LICENSE](../../LICENSE)
