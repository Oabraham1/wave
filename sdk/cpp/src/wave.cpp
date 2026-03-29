// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

/// WAVE C/C++ SDK implementation.
///
/// Implements the C API defined in wave.h. For v1, calls wave-compiler and
/// wave-emu as subprocesses. Direct FFI to the Rust runtime is planned for v2.

#include "wave/wave.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

static thread_local std::string g_last_error;

static void set_error(const std::string &msg) { g_last_error = msg; }

struct wave_device {
    wave_vendor_t vendor;
    std::string name;
};

struct wave_buffer {
    std::vector<uint8_t> data;
    size_t count;
    wave_dtype_t dtype;
};

struct wave_kernel {
    std::vector<uint8_t> wbin;
};

static std::string find_tool(const char *name) {
    std::string candidates[] = {
        std::string("target/release/") + name,
        std::string("target/debug/") + name,
        std::string("../../target/release/") + name,
        std::string("../../target/debug/") + name,
    };

    for (const auto &path : candidates) {
        FILE *f = fopen(path.c_str(), "r");
        if (f) {
            fclose(f);
            return path;
        }
    }

    return std::string(name);
}

wave_device_t *wave_detect_device(void) {
    auto *dev = new wave_device;

#ifdef __APPLE__
    dev->vendor = WAVE_VENDOR_APPLE;
    dev->name = "Apple GPU (Metal)";
    return dev;
#endif

    if (system("nvidia-smi > /dev/null 2>&1") == 0) {
        dev->vendor = WAVE_VENDOR_NVIDIA;
        dev->name = "NVIDIA GPU (CUDA)";
        return dev;
    }

    if (system("rocminfo > /dev/null 2>&1") == 0) {
        dev->vendor = WAVE_VENDOR_AMD;
        dev->name = "AMD GPU (ROCm)";
        return dev;
    }

    if (system("sycl-ls > /dev/null 2>&1") == 0) {
        dev->vendor = WAVE_VENDOR_INTEL;
        dev->name = "Intel GPU (SYCL)";
        return dev;
    }

    dev->vendor = WAVE_VENDOR_EMULATOR;
    dev->name = "WAVE Emulator (no GPU)";
    return dev;
}

wave_vendor_t wave_device_vendor(const wave_device_t *dev) { return dev->vendor; }

const char *wave_device_name(const wave_device_t *dev) { return dev->name.c_str(); }

void wave_free_device(wave_device_t *dev) { delete dev; }

wave_buffer_t *wave_create_buffer_f32(const float *data, size_t count) {
    auto *buf = new wave_buffer;
    buf->count = count;
    buf->dtype = WAVE_DTYPE_F32;
    buf->data.resize(count * sizeof(float));
    memcpy(buf->data.data(), data, count * sizeof(float));
    return buf;
}

wave_buffer_t *wave_create_zeros_f32(size_t count) {
    auto *buf = new wave_buffer;
    buf->count = count;
    buf->dtype = WAVE_DTYPE_F32;
    buf->data.resize(count * sizeof(float), 0);
    return buf;
}

int wave_read_buffer_f32(const wave_buffer_t *buf, float *out, size_t count) {
    if (buf->dtype != WAVE_DTYPE_F32) {
        set_error("buffer is not f32");
        return -1;
    }
    if (count > buf->count) {
        set_error("read count exceeds buffer size");
        return -1;
    }
    memcpy(out, buf->data.data(), count * sizeof(float));
    return 0;
}

void wave_free_buffer(wave_buffer_t *buf) { delete buf; }

wave_kernel_t *wave_compile(const char *source, wave_lang_t lang) {
    std::string compiler = find_tool("wave-compiler");
    const char *lang_str = "python";
    const char *ext = ".py";

    switch (lang) {
    case WAVE_LANG_PYTHON:
        lang_str = "python";
        ext = ".py";
        break;
    case WAVE_LANG_RUST:
        lang_str = "rust";
        ext = ".rs";
        break;
    case WAVE_LANG_CPP:
        lang_str = "cpp";
        ext = ".cpp";
        break;
    case WAVE_LANG_TYPESCRIPT:
        lang_str = "typescript";
        ext = ".ts";
        break;
    }

    char src_path[256];
    snprintf(src_path, sizeof(src_path), "/tmp/wave_kernel_XXXXXX%s", ext);
    int fd = mkstemps(src_path, static_cast<int>(strlen(ext)));
    if (fd < 0) {
        set_error("failed to create temp file");
        return nullptr;
    }

    FILE *f = fdopen(fd, "w");
    fprintf(f, "%s", source);
    fclose(f);

    char wbin_path[280];
    snprintf(wbin_path, sizeof(wbin_path), "%s.wbin", src_path);

    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "%s %s -o %s -l %s 2>&1", compiler.c_str(), src_path, wbin_path,
             lang_str);

    FILE *pipe = popen(cmd, "r");
    if (!pipe) {
        set_error("failed to run compiler");
        remove(src_path);
        return nullptr;
    }

    char output[4096] = {0};
    size_t bytes_read = fread(output, 1, sizeof(output) - 1, pipe);
    output[bytes_read] = '\0';
    int ret = pclose(pipe);

    remove(src_path);

    if (ret != 0) {
        set_error(std::string("compilation failed: ") + output);
        remove(wbin_path);
        return nullptr;
    }

    f = fopen(wbin_path, "rb");
    if (!f) {
        set_error("failed to read wbin output");
        return nullptr;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    auto *kern = new wave_kernel;
    kern->wbin.resize(static_cast<size_t>(size));
    size_t read = fread(kern->wbin.data(), 1, static_cast<size_t>(size), f);
    fclose(f);
    remove(wbin_path);

    if (static_cast<long>(read) != size) {
        set_error("failed to read complete wbin file");
        delete kern;
        return nullptr;
    }

    return kern;
}

int wave_launch(wave_kernel_t *kern, const wave_device_t *dev, wave_buffer_t **buffers,
                size_t n_buffers, const uint32_t *scalars, size_t n_scalars, const uint32_t grid[3],
                const uint32_t workgroup[3]) {
    if (dev->vendor != WAVE_VENDOR_EMULATOR) {
        set_error("only emulator launch is supported in v1 C/C++ SDK");
        return -1;
    }

    std::string emulator = find_tool("wave-emu");

    char tmpdir[] = "/tmp/wave_launch_XXXXXX";
    if (!mkdtemp(tmpdir)) {
        set_error("failed to create temp directory");
        return -1;
    }

    char wbin_path[280];
    snprintf(wbin_path, sizeof(wbin_path), "%s/kernel.wbin", tmpdir);
    FILE *f = fopen(wbin_path, "wb");
    fwrite(kern->wbin.data(), 1, kern->wbin.size(), f);
    fclose(f);

    char mem_path[280];
    snprintf(mem_path, sizeof(mem_path), "%s/memory.bin", tmpdir);
    f = fopen(mem_path, "wb");

    std::vector<uint32_t> offsets;
    uint32_t offset = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        offsets.push_back(offset);
        fwrite(buffers[i]->data.data(), 1, buffers[i]->data.size(), f);
        offset += static_cast<uint32_t>(buffers[i]->data.size());
    }
    fclose(f);

    std::string cmd = emulator + " " + wbin_path + " --memory-file " + mem_path + " --grid " +
                      std::to_string(grid[0]) + "," + std::to_string(grid[1]) + "," +
                      std::to_string(grid[2]) + " --workgroup " + std::to_string(workgroup[0]) +
                      "," + std::to_string(workgroup[1]) + "," + std::to_string(workgroup[2]);

    for (size_t i = 0; i < n_buffers; i++) {
        cmd += " --reg " + std::to_string(i) + "=" + std::to_string(offsets[i]);
    }
    for (size_t i = 0; i < n_scalars; i++) {
        cmd += " --reg " + std::to_string(n_buffers + i) + "=" + std::to_string(scalars[i]);
    }

    cmd += " 2>&1";
    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        set_error("failed to run emulator");
        return -1;
    }

    char output[4096] = {0};
    size_t bytes_read = fread(output, 1, sizeof(output) - 1, pipe);
    output[bytes_read] = '\0';
    int ret = pclose(pipe);

    if (ret != 0) {
        set_error(std::string("emulator failed: ") + output);
        return -1;
    }

    f = fopen(mem_path, "rb");
    if (!f) {
        set_error("failed to read emulator output memory");
        return -1;
    }

    offset = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        fseek(f, offset, SEEK_SET);
        size_t sz = buffers[i]->data.size();
        size_t read = fread(buffers[i]->data.data(), 1, sz, f);
        if (read != sz) {
            fclose(f);
            set_error("failed to read buffer output");
            return -1;
        }
        offset += static_cast<uint32_t>(sz);
    }
    fclose(f);

    remove(wbin_path);
    remove(mem_path);
    rmdir(tmpdir);

    return 0;
}

void wave_free_kernel(wave_kernel_t *kern) { delete kern; }

const char *wave_last_error(void) {
    if (g_last_error.empty()) {
        return nullptr;
    }
    return g_last_error.c_str();
}
