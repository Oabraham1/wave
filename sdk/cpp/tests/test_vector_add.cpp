// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

/// End-to-end test for the WAVE C/C++ SDK.

#include "wave/wave.h"

#include <cassert>
#include <cmath>
#include <cstdio>

static void test_device_detection() {
    wave_device_t *dev = wave_detect_device();
    assert(dev != nullptr);

    wave_vendor_t vendor = wave_device_vendor(dev);
    assert(vendor >= WAVE_VENDOR_APPLE && vendor <= WAVE_VENDOR_EMULATOR);

    const char *name = wave_device_name(dev);
    assert(name != nullptr);
    assert(name[0] != '\0');

    printf("Device: %s\n", name);
    wave_free_device(dev);
}

static void test_buffer_f32() {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    wave_buffer_t *buf = wave_create_buffer_f32(data, 4);
    assert(buf != nullptr);

    float out[4] = {0};
    int ret = wave_read_buffer_f32(buf, out, 4);
    assert(ret == 0);
    assert(out[0] == 1.0f);
    assert(out[1] == 2.0f);
    assert(out[2] == 3.0f);
    assert(out[3] == 4.0f);

    wave_free_buffer(buf);
}

static void test_zeros_f32() {
    wave_buffer_t *buf = wave_create_zeros_f32(8);
    assert(buf != nullptr);

    float out[8] = {0};
    int ret = wave_read_buffer_f32(buf, out, 8);
    assert(ret == 0);
    for (int i = 0; i < 8; i++) {
        assert(out[i] == 0.0f);
    }

    wave_free_buffer(buf);
}

int main() {
    printf("Running WAVE C/C++ SDK tests...\n");

    test_device_detection();
    printf("  [PASS] device detection\n");

    test_buffer_f32();
    printf("  [PASS] buffer f32 roundtrip\n");

    test_zeros_f32();
    printf("  [PASS] zeros f32\n");

    printf("All C/C++ SDK tests passed!\n");
    return 0;
}
