#include <cuda_runtime.h>

struct Memory {
    unsigned n;
    unsigned* a;
    unsigned* b;
    unsigned* output;
};

__device__ Memory memory_view(unsigned* memory) {
    auto* bytes = reinterpret_cast<unsigned char*>(memory);
    return {memory[0], reinterpret_cast<unsigned*>(bytes + memory[1]),
            reinterpret_cast<unsigned*>(bytes + memory[2]),
            reinterpret_cast<unsigned*>(bytes + memory[3])};
}

extern "C" __global__ void vector_add(unsigned* memory) {
    const auto view = memory_view(memory);
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < view.n) {
        view.output[i] = __float_as_uint(__uint_as_float(view.a[i]) +
                                         __uint_as_float(view.b[i]));
    }
}

extern "C" __global__ void divergent(unsigned* memory) {
    const auto view = memory_view(memory);
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < view.n) {
        const float x = __uint_as_float(view.a[i]);
        view.output[i] = __float_as_uint(x < 0.0f ? x - 1.0f : x * 2.0f);
    }
}

extern "C" __global__ void gather(unsigned* memory) {
    const auto view = memory_view(memory);
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < view.n) {
        view.output[i] = view.a[view.b[i]];
    }
}

__device__ void histogram(unsigned* memory) {
    const auto view = memory_view(memory);
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < view.n) {
        atomicAdd(view.output + view.a[i], 1u);
    }
}

extern "C" __global__ void histogram_uniform(unsigned* memory) {
    histogram(memory);
}

extern "C" __global__ void histogram_skewed(unsigned* memory) {
    histogram(memory);
}

extern "C" __global__ void reduce_shuffle(unsigned* memory) {
    const auto view = memory_view(memory);
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    float value = i < view.n ? __uint_as_float(view.a[i]) : 0.0f;
    #pragma unroll
    for (unsigned offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    }
    if ((threadIdx.x & 31u) == 0) {
        view.output[i / 32] = __float_as_uint(value);
    }
}

extern "C" __global__ void reduce_shared(unsigned* memory) {
    const auto view = memory_view(memory);
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned t = threadIdx.x;
    __shared__ float scratch[128];
    float value = i < view.n ? __uint_as_float(view.a[i]) : 0.0f;
    #pragma unroll
    for (unsigned offset = 16; offset > 0; offset >>= 1) {
        scratch[t] = value;
        __syncthreads();
        const float other = scratch[t ^ offset];
        __syncthreads();
        value += other;
    }
    if ((t & 31u) == 0) {
        view.output[i / 32] = __float_as_uint(value);
    }
}

extern "C" __global__ void csr_spmv(unsigned* memory) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < memory[0]) {
        auto* bytes = reinterpret_cast<unsigned char*>(memory);
        const auto* row_ptr = reinterpret_cast<const unsigned*>(bytes + memory[1]);
        const auto* columns = reinterpret_cast<const unsigned*>(bytes + memory[2]);
        auto* output = reinterpret_cast<float*>(bytes + memory[3]);
        const auto* values = reinterpret_cast<const float*>(bytes + memory[4]);
        const auto* x = reinterpret_cast<const float*>(bytes + memory[5]);
        float sum = 0.0f;
        for (unsigned j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
            sum += values[j] * x[columns[j]];
        }
        output[row] = sum;
    }
}

extern "C" __global__ void stencil_2d(unsigned* memory) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < memory[0]) {
        auto* bytes = reinterpret_cast<unsigned char*>(memory);
        const auto* input = reinterpret_cast<const float*>(bytes + memory[1]);
        const unsigned width = memory[2];
        auto* output = reinterpret_cast<float*>(bytes + memory[3]);
        const unsigned height = memory[4];
        const unsigned x = i & (width - 1);
        const unsigned y = i >> memory[5];
        const float center = input[i];
        if (x == 0 || x + 1 == width || y == 0 || y + 1 == height) {
            output[i] = center;
        } else {
            const float neighbors = ((input[i - width] + input[i + width]) +
                                     input[i - 1]) + input[i + 1];
            output[i] = center * 0.5f + neighbors * 0.125f;
        }
    }
}
