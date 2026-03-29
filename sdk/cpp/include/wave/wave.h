/* Copyright 2026 Ojima Abraham */
/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file wave.h
 * @brief WAVE GPU SDK for C/C++ - write GPU kernels, run on any GPU.
 */

#ifndef WAVE_WAVE_H
#define WAVE_WAVE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** GPU vendor types. */
typedef enum {
    WAVE_VENDOR_APPLE = 0,
    WAVE_VENDOR_NVIDIA = 1,
    WAVE_VENDOR_AMD = 2,
    WAVE_VENDOR_INTEL = 3,
    WAVE_VENDOR_EMULATOR = 4,
} wave_vendor_t;

/** Source language types. */
typedef enum {
    WAVE_LANG_PYTHON = 0,
    WAVE_LANG_RUST = 1,
    WAVE_LANG_CPP = 2,
    WAVE_LANG_TYPESCRIPT = 3,
} wave_lang_t;

/** Element data types. */
typedef enum {
    WAVE_DTYPE_F32 = 0,
    WAVE_DTYPE_U32 = 1,
    WAVE_DTYPE_I32 = 2,
    WAVE_DTYPE_F16 = 3,
    WAVE_DTYPE_F64 = 4,
} wave_dtype_t;

/** Opaque handle to a detected GPU device. */
typedef struct wave_device wave_device_t;

/** Opaque handle to a device memory buffer. */
typedef struct wave_buffer wave_buffer_t;

/** Opaque handle to a compiled kernel. */
typedef struct wave_kernel wave_kernel_t;

/** Detect the best available GPU device.
 *
 *  @return Device handle, or NULL on error. Must be freed with wave_free_device().
 */
wave_device_t *wave_detect_device(void);

/** Get the vendor of a device. */
wave_vendor_t wave_device_vendor(const wave_device_t *dev);

/** Get the name of a device.
 *
 *  @return Null-terminated string. Valid as long as the device handle is alive.
 */
const char *wave_device_name(const wave_device_t *dev);

/** Free a device handle. */
void wave_free_device(wave_device_t *dev);

/** Create a buffer from float data.
 *
 *  @param data  Pointer to float array.
 *  @param count Number of elements.
 *  @return Buffer handle, or NULL on error. Must be freed with wave_free_buffer().
 */
wave_buffer_t *wave_create_buffer_f32(const float *data, size_t count);

/** Create a zero-filled float buffer.
 *
 *  @param count Number of elements.
 *  @return Buffer handle, or NULL on error.
 */
wave_buffer_t *wave_create_zeros_f32(size_t count);

/** Read float data from a buffer.
 *
 *  @param buf   Buffer handle.
 *  @param out   Output array (must have space for count elements).
 *  @param count Number of elements to read.
 *  @return 0 on success, -1 on error.
 */
int wave_read_buffer_f32(const wave_buffer_t *buf, float *out, size_t count);

/** Free a buffer handle. */
void wave_free_buffer(wave_buffer_t *buf);

/** Compile kernel source code.
 *
 *  @param source Null-terminated kernel source string.
 *  @param lang   Source language.
 *  @return Kernel handle, or NULL on error. Must be freed with wave_free_kernel().
 */
wave_kernel_t *wave_compile(const char *source, wave_lang_t lang);

/** Launch a compiled kernel.
 *
 *  @param kern       Compiled kernel handle.
 *  @param dev        Device to run on.
 *  @param buffers    Array of buffer handles.
 *  @param n_buffers  Number of buffers.
 *  @param scalars    Array of scalar uint32 values.
 *  @param n_scalars  Number of scalars.
 *  @param grid       Grid dimensions [x, y, z].
 *  @param workgroup  Workgroup dimensions [x, y, z].
 *  @return 0 on success, -1 on error.
 */
int wave_launch(wave_kernel_t *kern, const wave_device_t *dev, wave_buffer_t **buffers,
                size_t n_buffers, const uint32_t *scalars, size_t n_scalars, const uint32_t grid[3],
                const uint32_t workgroup[3]);

/** Free a kernel handle. */
void wave_free_kernel(wave_kernel_t *kern);

/** Get the last error message.
 *
 *  @return Null-terminated error string, or NULL if no error.
 *          Valid until the next WAVE API call.
 */
const char *wave_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* WAVE_WAVE_H */
