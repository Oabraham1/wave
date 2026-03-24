// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

// SYCL host program that includes a WAVE-compiled SYCL kernel, allocates USM device
// memory with test data, dispatches the kernel on an Intel GPU, and reads back
// results. Demonstrates end-to-end execution of a WAVE program on Intel hardware
// via oneAPI/SYCL.
//
// Build: icpx -fsycl -O3 -DKERNEL_FILE=\"program.cpp\" run_on_intel.cpp -o runner
// Run:   ./runner

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

using namespace sycl;

#ifndef KERNEL_FILE
#define KERNEL_FILE "vector_add.cpp"
#endif

#include KERNEL_FILE

int main(int argc, char** argv) {
    int workgroup_count = (argc >= 2) ? atoi(argv[1]) : 1;
    int element_count = 256 * workgroup_count;
    size_t buf_size = element_count * sizeof(float) * 3;

    queue q{gpu_selector_v};

    auto dev = q.get_device();
    printf("Device: %s\n", dev.get_info<info::device::name>().c_str());
    printf("Max compute units: %d\n", dev.get_info<info::device::max_compute_units>());

    uint8_t* d_mem = malloc_device<uint8_t>(buf_size, q);
    q.memset(d_mem, 0, buf_size).wait();

    std::vector<float> h_buf(element_count * 3);
    for (int i = 0; i < element_count; i++) {
        h_buf[i] = (float)i;
        h_buf[element_count + i] = (float)i * 2.0f;
        h_buf[element_count * 2 + i] = 0.0f;
    }
    q.memcpy(d_mem, h_buf.data(), buf_size).wait();

    vector_add_launch(q, d_mem,
                      workgroup_count, 1, 1,
                      256, 1, 1);

    q.memcpy(h_buf.data(), d_mem, buf_size).wait();

    float* output = h_buf.data() + element_count * 2;
    printf("Results (first 16 elements):\n");
    for (int i = 0; i < 16 && i < element_count; i++) {
        printf("  c[%d] = %f\n", i, output[i]);
    }

    free(d_mem, q);
    return 0;
}
