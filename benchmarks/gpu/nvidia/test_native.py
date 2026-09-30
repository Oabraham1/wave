"""Check scalar CUDA workload algorithms on the CPU with launch intrinsics stubbed."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("gpu_workloads", HERE.parent / "workloads.py")
WORKLOADS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKLOADS)

HEADER = r"""
#pragma once
#include <cstring>
#define __device__
#define __global__
#define __shared__
struct Dimension { unsigned x = 0; };
inline Dimension blockIdx, blockDim, threadIdx;
inline unsigned __float_as_uint(float value) {
    unsigned result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
inline float __uint_as_float(unsigned value) {
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
inline void atomicAdd(unsigned* value, unsigned amount) { *value += amount; }
inline float __shfl_xor_sync(unsigned, float value, unsigned, unsigned) { return value; }
inline void __syncthreads() {}
"""

WRAPPER = r"""
#include <fstream>
#include <string>
#include <vector>
#include "native.cu"

int main(int argc, char** argv) {
    if (argc != 4) return 1;
    std::ifstream input(argv[2], std::ios::binary | std::ios::ate);
    if (!input) return 2;
    const auto size = input.tellg();
    if (size <= 0 || size % 4 != 0) return 3;
    std::vector<unsigned> memory(static_cast<std::size_t>(size) / 4);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(memory.data()), size);
    if (!input) return 4;
    blockDim.x = 128;
    for (blockIdx.x = 0; blockIdx.x < (memory[0] + 127) / 128; ++blockIdx.x) {
        for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x) {
            if (std::string(argv[1]) == "csr_spmv") csr_spmv(memory.data());
            else if (std::string(argv[1]) == "stencil_2d") stencil_2d(memory.data());
            else return 5;
        }
    }
    std::ofstream output(argv[3], std::ios::binary);
    output.write(reinterpret_cast<const char*>(memory.data()), size);
    return output ? 0 : 6;
}
"""


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("a C++ compiler is required for the scalar algorithm check")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        (cls.work / "cuda_runtime.h").write_text(HEADER)
        (cls.work / "wrapper.cpp").write_text(WRAPPER)
        cls.executable = cls.work / "native-cpu-check"
        subprocess.run([compiler, "-std=c++17", "-O2", "-ffp-contract=off", "-fno-strict-aliasing",
                        "-I", str(cls.work), "-I", str(HERE), str(cls.work / "wrapper.cpp"),
                        "-o", str(cls.executable)], check=True, capture_output=True, text=True)

    def test_application_memory_matches_independent_reference(self):
        for name in ("csr_spmv", "stencil_2d"):
            for size in (1, 3, 17, 129, 1025):
                for seed in (17, 71):
                    with self.subTest(name=name, size=size, seed=seed):
                        initial, expected, _ = WORKLOADS.make_case(name, size, seed)
                        input_path, output_path = self.work / "input.bin", self.work / "output.bin"
                        input_path.write_bytes(initial)
                        subprocess.run([str(self.executable), name, str(input_path), str(output_path)],
                                       check=True)
                        self.assertEqual(output_path.read_bytes(), expected)


if __name__ == "__main__":
    unittest.main()
