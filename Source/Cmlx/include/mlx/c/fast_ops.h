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

// ============================================================================
// Existing ops
// ============================================================================

void mlx_fast_matmul(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr);

void mlx_fast_reshape(
    void* result_ptr,
    const void* a_ptr,
    const int32_t* shape,
    size_t shape_num,
    const void* stream_ptr);

void mlx_fast_transpose(
    void* result_ptr,
    const void* a_ptr,
    const int* axes,
    size_t axes_num,
    const void* stream_ptr);

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

void mlx_fast_add(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr);

void mlx_fast_multiply(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr);

void mlx_fast_softmax(
    void* result_ptr,
    const void* a_ptr,
    const int* axes,
    size_t axes_num,
    bool precise,
    const void* stream_ptr);

// ============================================================================
// Element-wise binary ops
// ============================================================================

void mlx_fast_subtract(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr);

void mlx_fast_divide(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr);

void mlx_fast_maximum(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr);

void mlx_fast_minimum(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr);

void mlx_fast_power(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr);

void mlx_fast_where(
    void* result_ptr,
    const void* condition_ptr,
    const void* x_ptr,
    const void* y_ptr,
    const void* stream_ptr);

// ============================================================================
// Element-wise unary ops
// ============================================================================

void mlx_fast_negative(
    void* result_ptr,
    const void* a_ptr,
    const void* stream_ptr);

void mlx_fast_abs(
    void* result_ptr,
    const void* a_ptr,
    const void* stream_ptr);

void mlx_fast_exp(
    void* result_ptr,
    const void* a_ptr,
    const void* stream_ptr);

void mlx_fast_log(
    void* result_ptr,
    const void* a_ptr,
    const void* stream_ptr);

void mlx_fast_sigmoid(
    void* result_ptr,
    const void* a_ptr,
    const void* stream_ptr);

void mlx_fast_logical_not(
    void* result_ptr,
    const void* a_ptr,
    const void* stream_ptr);

/** silu(x) = x * sigmoid(x). Composed from MLX primitives. */
void mlx_fast_silu(
    void* result_ptr,
    const void* a_ptr,
    const void* stream_ptr);

/** log_softmax(x, axes) = x - logsumexp(x, axes). */
void mlx_fast_log_softmax(
    void* result_ptr,
    const void* a_ptr,
    const int* axes,
    size_t axes_num,
    const void* stream_ptr);

// ============================================================================
// Comparison ops
// ============================================================================

void mlx_fast_equal(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr);

void mlx_fast_less(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr);

void mlx_fast_greater(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr);

void mlx_fast_greater_equal(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr);

void mlx_fast_less_equal(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr);

// ============================================================================
// Reduction ops
// ============================================================================

void mlx_fast_sum_axis(
    void* result_ptr,
    const void* a_ptr,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const void* stream_ptr);

void mlx_fast_sum_all(
    void* result_ptr,
    const void* a_ptr,
    bool keepdims,
    const void* stream_ptr);

void mlx_fast_mean_axis(
    void* result_ptr,
    const void* a_ptr,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const void* stream_ptr);

void mlx_fast_mean_all(
    void* result_ptr,
    const void* a_ptr,
    bool keepdims,
    const void* stream_ptr);

void mlx_fast_max_axis(
    void* result_ptr,
    const void* a_ptr,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const void* stream_ptr);

void mlx_fast_max_all(
    void* result_ptr,
    const void* a_ptr,
    bool keepdims,
    const void* stream_ptr);

void mlx_fast_min_axis(
    void* result_ptr,
    const void* a_ptr,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const void* stream_ptr);

void mlx_fast_min_all(
    void* result_ptr,
    const void* a_ptr,
    bool keepdims,
    const void* stream_ptr);

void mlx_fast_argmax_axis(
    void* result_ptr,
    const void* a_ptr,
    int axis,
    bool keepdims,
    const void* stream_ptr);

void mlx_fast_argmax_all(
    void* result_ptr,
    const void* a_ptr,
    bool keepdims,
    const void* stream_ptr);

void mlx_fast_argmin_axis(
    void* result_ptr,
    const void* a_ptr,
    int axis,
    bool keepdims,
    const void* stream_ptr);

void mlx_fast_argmin_all(
    void* result_ptr,
    const void* a_ptr,
    bool keepdims,
    const void* stream_ptr);

// ============================================================================
// Shape / indexing ops
// ============================================================================

/**
 * Concatenate arrays along an axis.
 * arrays_ptr is an array of void* (each a mlx::core::array*).
 */
void mlx_fast_concatenate(
    void* result_ptr,
    const void* const* arrays_ptr,
    size_t num_arrays,
    int axis,
    const void* stream_ptr);

/**
 * Split into num_splits equal parts along axis.
 * results_ptr must have space for num_splits pre-allocated arrays.
 */
void mlx_fast_split_equal(
    void** results_ptr,
    size_t* num_results,
    const void* a_ptr,
    int num_splits,
    int axis,
    const void* stream_ptr);

/**
 * Split at given indices along axis.
 * results_ptr must have space for (indices_num + 1) pre-allocated arrays.
 */
void mlx_fast_split_indices(
    void** results_ptr,
    size_t* num_results,
    const void* a_ptr,
    const int32_t* indices,
    size_t indices_num,
    int axis,
    const void* stream_ptr);

void mlx_fast_expand_dims(
    void* result_ptr,
    const void* a_ptr,
    const int* axes,
    size_t axes_num,
    const void* stream_ptr);

void mlx_fast_expand_dims_axis(
    void* result_ptr,
    const void* a_ptr,
    int axis,
    const void* stream_ptr);

void mlx_fast_squeeze_axes(
    void* result_ptr,
    const void* a_ptr,
    const int* axes,
    size_t axes_num,
    const void* stream_ptr);

void mlx_fast_squeeze_axis(
    void* result_ptr,
    const void* a_ptr,
    int axis,
    const void* stream_ptr);

void mlx_fast_squeeze_all(
    void* result_ptr,
    const void* a_ptr,
    const void* stream_ptr);

/**
 * Cast array to dtype. dtype_val and dtype_size correspond to
 * mlx::core::Dtype::Val and size fields.
 */
void mlx_fast_astype(
    void* result_ptr,
    const void* a_ptr,
    uint8_t dtype_val,
    uint8_t dtype_size,
    const void* stream_ptr);

void mlx_fast_flatten(
    void* result_ptr,
    const void* a_ptr,
    int start_axis,
    int end_axis,
    const void* stream_ptr);

void mlx_fast_flatten_all(
    void* result_ptr,
    const void* a_ptr,
    const void* stream_ptr);

void mlx_fast_take_axis(
    void* result_ptr,
    const void* a_ptr,
    const void* indices_ptr,
    int axis,
    const void* stream_ptr);

void mlx_fast_take(
    void* result_ptr,
    const void* a_ptr,
    const void* indices_ptr,
    const void* stream_ptr);

void mlx_fast_take_along_axis(
    void* result_ptr,
    const void* a_ptr,
    const void* indices_ptr,
    int axis,
    const void* stream_ptr);

void mlx_fast_put_along_axis(
    void* result_ptr,
    const void* a_ptr,
    const void* indices_ptr,
    const void* values_ptr,
    int axis,
    const void* stream_ptr);

/**
 * Single-axis gather: gather(a, indices, axis, slice_sizes).
 */
void mlx_fast_gather(
    void* result_ptr,
    const void* a_ptr,
    const void* indices_ptr,
    int axis,
    const int32_t* slice_sizes,
    size_t slice_sizes_num,
    const void* stream_ptr);

// ============================================================================
// Sort / partition ops
// ============================================================================

void mlx_fast_sort_axis(
    void* result_ptr,
    const void* a_ptr,
    int axis,
    const void* stream_ptr);

void mlx_fast_sort_all(
    void* result_ptr,
    const void* a_ptr,
    const void* stream_ptr);

void mlx_fast_argsort_axis(
    void* result_ptr,
    const void* a_ptr,
    int axis,
    const void* stream_ptr);

void mlx_fast_argsort_all(
    void* result_ptr,
    const void* a_ptr,
    const void* stream_ptr);

void mlx_fast_argpartition_axis(
    void* result_ptr,
    const void* a_ptr,
    int kth,
    int axis,
    const void* stream_ptr);

void mlx_fast_argpartition_all(
    void* result_ptr,
    const void* a_ptr,
    int kth,
    const void* stream_ptr);

// ============================================================================
// Cumulative ops
// ============================================================================

void mlx_fast_cumsum_axis(
    void* result_ptr,
    const void* a_ptr,
    int axis,
    bool reverse,
    bool inclusive,
    const void* stream_ptr);

void mlx_fast_cumsum_all(
    void* result_ptr,
    const void* a_ptr,
    bool reverse,
    bool inclusive,
    const void* stream_ptr);

// ============================================================================
// Convolution
// ============================================================================

void mlx_fast_conv1d(
    void* result_ptr,
    const void* input_ptr,
    const void* weight_ptr,
    int stride,
    int padding,
    int dilation,
    int groups,
    const void* stream_ptr);

// ============================================================================
// Array creation
// ============================================================================

void mlx_fast_zeros(
    void* result_ptr,
    const int32_t* shape,
    size_t shape_num,
    uint8_t dtype_val,
    uint8_t dtype_size,
    const void* stream_ptr);

void mlx_fast_ones(
    void* result_ptr,
    const int32_t* shape,
    size_t shape_num,
    uint8_t dtype_val,
    uint8_t dtype_size,
    const void* stream_ptr);

void mlx_fast_full_float(
    void* result_ptr,
    const int32_t* shape,
    size_t shape_num,
    float val,
    uint8_t dtype_val,
    uint8_t dtype_size,
    const void* stream_ptr);

void mlx_fast_arange_float(
    void* result_ptr,
    double start,
    double stop,
    double step,
    uint8_t dtype_val,
    uint8_t dtype_size,
    const void* stream_ptr);

void mlx_fast_arange_int(
    void* result_ptr,
    int start,
    int stop,
    int step,
    const void* stream_ptr);

// ============================================================================
// Utility
// ============================================================================

void* mlx_fast_alloc_array(void);
void mlx_fast_free_array(void* arr_ptr);
const void* mlx_fast_default_gpu_stream(void);
const void* mlx_fast_default_cpu_stream(void);

#ifdef __cplusplus
}
#endif

#endif /* MLX_FAST_OPS_H */
