// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

// HIP host program that includes a WAVE-compiled HIP kernel, allocates device memory
// with test data, dispatches the kernel on an AMD GPU, and reads back results.
// Demonstrates end-to-end execution of a WAVE program on AMD hardware via ROCm/HIP.
//
// Build: hipcc -O3 -DKERNEL_FILE=\"program.hip\" run_on_amd.hip -o runner
// Run:   ./runner
//
// Or include the kernel directly by editing the #include below.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>

#ifndef KERNEL_FILE
#define KERNEL_FILE "vector_add.hip"
#endif

#include KERNEL_FILE

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t err = (call);                                               \
        if (err != hipSuccess) {                                               \
            fprintf(stderr, "HIP error %d at %s:%d\n", err, __FILE__,         \
                    __LINE__);                                                 \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

int main(int argc, char** argv) {
    int workgroup_count = (argc >= 2) ? atoi(argv[1]) : 1;
    int element_count = 256 * workgroup_count;
    size_t buf_size = element_count * sizeof(float) * 3;

    uint8_t* d_mem;
    HIP_CHECK(hipMalloc(&d_mem, buf_size));
    HIP_CHECK(hipMemset(d_mem, 0, buf_size));

    float* h_buf = (float*)malloc(buf_size);
    for (int i = 0; i < element_count; i++) {
        h_buf[i] = (float)i;
        h_buf[element_count + i] = (float)i * 2.0f;
        h_buf[element_count * 2 + i] = 0.0f;
    }
    HIP_CHECK(hipMemcpy(d_mem, h_buf, buf_size, hipMemcpyHostToDevice));

    dim3 grid(workgroup_count, 1, 1);
    dim3 block(256, 1, 1);
    size_t shared_mem = 16384;

    hipLaunchKernelGGL(vector_add, grid, block, shared_mem, 0, d_mem);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_buf, d_mem, buf_size, hipMemcpyDeviceToHost));

    float* output = h_buf + element_count * 2;
    printf("Results (first 16 elements):\n");
    for (int i = 0; i < 16 && i < element_count; i++) {
        printf("  c[%d] = %f\n", i, output[i]);
    }

    free(h_buf);
    HIP_CHECK(hipFree(d_mem));
    return 0;
}
