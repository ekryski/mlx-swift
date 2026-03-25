/* Copyright © 2026 Eric Kryski */
/* Thin direct-call wrappers for hot-path MLX operations.               */
/* These bypass the full mlx-c bridge overhead (null checks, heap       */
/* allocation, try/catch per call) by operating directly on the raw     */
/* void* ctx pointers that mlx_array already stores.                    */
/*                                                                       */
/* SAFETY: callers MUST ensure pointers are valid mlx::core::array*     */
/* (or mlx::core::Stream* for stream). No null checks are performed.    */

#ifndef MLX_FAST_OPS_H
#define MLX_FAST_OPS_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Direct matmul: result = a @ b.
 * All pointers are raw mlx::core::array* cast to void*.
 * result_ptr must point to a live mlx::core::array (will be overwritten).
 * stream_ptr is a raw mlx::core::Stream*.
 */
void mlx_fast_matmul(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr);

/**
 * Direct reshape: result = reshape(a, shape).
 * shape is an array of int32_t with shape_num elements.
 */
void mlx_fast_reshape(
    void* result_ptr,
    const void* a_ptr,
    const int32_t* shape,
    size_t shape_num,
    const void* stream_ptr);

/**
 * Direct transpose: result = transpose(a, axes).
 * axes is an array of int with axes_num elements.
 */
void mlx_fast_transpose(
    void* result_ptr,
    const void* a_ptr,
    const int* axes,
    size_t axes_num,
    const void* stream_ptr);

/**
 * Direct quantized_matmul: result = quantized_matmul(x, w, scales, biases).
 * biases_ptr may be NULL to indicate no biases.
 */
void mlx_fast_quantized_matmul(
    void* result_ptr,
    const void* x_ptr,
    const void* w_ptr,
    const void* scales_ptr,
    const void* biases_ptr,
    bool transpose,
    int group_size,
    int bits,
    const void* stream_ptr);

/**
 * Direct add: result = a + b.
 */
void mlx_fast_add(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr);

/**
 * Direct multiply: result = a * b.
 */
void mlx_fast_multiply(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr);

/**
 * Direct softmax: result = softmax(a, axes).
 */
void mlx_fast_softmax(
    void* result_ptr,
    const void* a_ptr,
    const int* axes,
    size_t axes_num,
    bool precise,
    const void* stream_ptr);

/**
 * Allocate a new mlx::core::array on the heap (default-constructed).
 * Returns a void* that can be used as result_ptr in the above functions.
 * Must be freed with mlx_fast_free_array.
 */
void* mlx_fast_alloc_array(void);

/**
 * Free an array allocated with mlx_fast_alloc_array.
 */
void mlx_fast_free_array(void* arr_ptr);

/**
 * Get the default GPU stream pointer (cached).
 * Returns a void* pointing to a static mlx::core::Stream.
 */
const void* mlx_fast_default_gpu_stream(void);

/**
 * Get the default CPU stream pointer (cached).
 * Returns a void* pointing to a static mlx::core::Stream.
 */
const void* mlx_fast_default_cpu_stream(void);

#ifdef __cplusplus
}
#endif

#endif /* MLX_FAST_OPS_H */
