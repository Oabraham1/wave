"""Matched Slang kernels for GPU compiler comparisons."""

from pathlib import Path
import subprocess

VERSION = "2026.19"
RELEASE_URL = "https://github.com/shader-slang/slang/releases/download/v2026.19/slang-2026.19-macos-aarch64.tar.gz"
RELEASE_SHA256 = "fa46822c92d81404a7951de3746093985c1e6f520f4b533e2aa3bea576f0f2f0"


def source(name):
    prefix = f'''[[vk::binding(0, 0)]] RWByteAddressBuffer memory;
groupshared float scratch[128];
[shader("compute")]
[numthreads(128, 1, 1)]
void {name}(uint3 tid : SV_DispatchThreadID, uint3 local : SV_GroupThreadID) {{
    uint i = tid.x;
    uint n = memory.Load(0);
    uint a = memory.Load(4);
    uint b = memory.Load(8);
    uint output = memory.Load(12);
'''
    if name == "vector_add":
        body = '''    if (i < n) {
        float x = asfloat(memory.Load(a + 4 * i));
        float y = asfloat(memory.Load(b + 4 * i));
        memory.Store(output + 4 * i, asuint(x + y));
    }
'''
    elif name == "divergent":
        body = '''    if (i < n) {
        float x = asfloat(memory.Load(a + 4 * i));
        float y = x < 0.0f ? x - 1.0f : x * 2.0f;
        memory.Store(output + 4 * i, asuint(y));
    }
'''
    elif name == "gather":
        body = '''    if (i < n) {
        uint index = memory.Load(b + 4 * i);
        memory.Store(output + 4 * i, memory.Load(a + 4 * index));
    }
'''
    elif name in ("histogram_uniform", "histogram_skewed"):
        body = '''    if (i < n) {
        uint bin = memory.Load(a + 4 * i);
        uint previous;
        memory.InterlockedAdd(output + 4 * bin, 1u, previous);
    }
'''
    elif name == "csr_spmv":
        body = '''    if (i < n) {
        uint values = memory.Load(16);
        uint vector = memory.Load(20);
        uint begin = memory.Load(a + 4 * i);
        uint end = memory.Load(a + 4 * (i + 1));
        float sum = 0.0f;
        for (uint j = begin; j < end; ++j) {
            uint column = memory.Load(b + 4 * j);
            float value = asfloat(memory.Load(values + 4 * j));
            float x = asfloat(memory.Load(vector + 4 * column));
            sum += value * x;
        }
        memory.Store(output + 4 * i, asuint(sum));
    }
'''
    elif name == "stencil_2d":
        body = '''    if (i < n) {
        uint width = b;
        uint height = memory.Load(16);
        uint x = i & (width - 1);
        uint y = i >> memory.Load(20);
        float value = asfloat(memory.Load(a + 4 * i));
        if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
            float north = asfloat(memory.Load(a + 4 * (i - width)));
            float south = asfloat(memory.Load(a + 4 * (i + width)));
            float west = asfloat(memory.Load(a + 4 * (i - 1)));
            float east = asfloat(memory.Load(a + 4 * (i + 1)));
            float neighbors = ((north + south) + west) + east;
            value = value * 0.5f + neighbors * 0.125f;
        }
        memory.Store(output + 4 * i, asuint(value));
    }
'''
    elif name in ("reduce_shuffle", "reduce_shared"):
        body = "    float value = i < n ? asfloat(memory.Load(a + 4 * i)) : 0.0f;\n"
        for offset in (16, 8, 4, 2, 1):
            if name == "reduce_shuffle":
                body += f"    value += WaveReadLaneAt(value, WaveGetLaneIndex() ^ {offset});\n"
            else:
                body += f'''    scratch[local.x] = value;
    AllMemoryBarrierWithGroupSync();
    float other{offset} = scratch[local.x ^ {offset}];
    value += other{offset};
    AllMemoryBarrierWithGroupSync();
'''
        body += "    if (WaveGetLaneIndex() == 0) memory.Store(output + 4 * (i / 32), asuint(value));\n"
    else:
        raise ValueError(f"no Slang baseline for {name}")
    return prefix + body + "}\n"


def compile_source(name, directory, compiler):
    """Compile an unmodified Slang source into the measured Metal input."""
    compiler = Path(compiler).resolve()
    reported = subprocess.run([str(compiler), "-version"], check=True, capture_output=True,
                              text=True)
    version = (reported.stdout + reported.stderr).strip()
    if version != VERSION:
        raise ValueError(f"expected Slang {VERSION}, found {version}")
    path = Path(directory) / f"{name}.slang"
    output = Path(directory) / f"{name}.slang.metal"
    path.write_text(source(name))
    command = [str(compiler), str(path), "-entry", name, "-stage", "compute", "-target",
               "metal", "-O3", "-fp-mode", "precise", "-o", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    path.with_suffix(".slang.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Slang failed for {name}: {result.stderr}")
    return {"version": version, "command": command, "source": path.name,
            "output": output.name, "returncode": result.returncode}
