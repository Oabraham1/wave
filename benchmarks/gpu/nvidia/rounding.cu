#include <cuda_runtime.h>

extern "C" __global__ void rounding_boundaries(unsigned* memory) {
    const float a = __uint_as_float(memory[0]);
    const float b = __uint_as_float(memory[1]);
    const float c = __uint_as_float(memory[2]);
    const float one = __uint_as_float(memory[3]);
    const float product = __fmul_rn(a, b);
    memory[4] = __float_as_uint(__fadd_rn(product, c));
    memory[5] = __float_as_uint(__fsub_rn(product, one));
    memory[6] = __float_as_uint(__fmaf_rn(a, b, c));
}
