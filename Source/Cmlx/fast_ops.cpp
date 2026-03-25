/* Copyright © 2026 Eric Kryski */
/* Thin direct-call wrappers for hot-path MLX operations.               */
/* Bypasses the full mlx-c bridge to eliminate per-operation overhead.   */

#include "mlx/mlx.h"

#include <optional>

// Cast helpers - no null checks, caller guarantees validity
static inline mlx::core::array& as_array(void* p) {
  return *static_cast<mlx::core::array*>(p);
}
static inline const mlx::core::array& as_const_array(const void* p) {
  return *static_cast<const mlx::core::array*>(p);
}
static inline const mlx::core::Stream& as_stream(const void* p) {
  return *static_cast<const mlx::core::Stream*>(p);
}

// Helper to convert dtype val+size to mlx::core::Dtype
static inline mlx::core::Dtype as_dtype(uint8_t val, uint8_t size) {
  return mlx::core::Dtype(static_cast<mlx::core::Dtype::Val>(val), size);
}

extern "C" {

// ============================================================================
// Existing ops
// ============================================================================

void mlx_fast_matmul(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::matmul(
      as_const_array(a_ptr),
      as_const_array(b_ptr),
      as_stream(stream_ptr));
}

void mlx_fast_reshape(
    void* result_ptr,
    const void* a_ptr,
    const int32_t* shape,
    size_t shape_num,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::reshape(
      as_const_array(a_ptr),
      mlx::core::Shape(shape, shape + shape_num),
      as_stream(stream_ptr));
}

void mlx_fast_transpose(
    void* result_ptr,
    const void* a_ptr,
    const int* axes,
    size_t axes_num,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::transpose(
      as_const_array(a_ptr),
      std::vector<int>(axes, axes + axes_num),
      as_stream(stream_ptr));
}

void mlx_fast_quantized_matmul(
    void* result_ptr,
    const void* x_ptr,
    const void* w_ptr,
    const void* scales_ptr,
    const void* biases_ptr,
    bool transpose,
    int group_size,
    int bits,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::quantized_matmul(
      as_const_array(x_ptr),
      as_const_array(w_ptr),
      as_const_array(scales_ptr),
      biases_ptr ? std::make_optional(as_const_array(biases_ptr))
                 : std::nullopt,
      transpose,
      group_size,
      bits,
      "affine",
      as_stream(stream_ptr));
}

void mlx_fast_add(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::add(
      as_const_array(a_ptr),
      as_const_array(b_ptr),
      as_stream(stream_ptr));
}

void mlx_fast_multiply(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::multiply(
      as_const_array(a_ptr),
      as_const_array(b_ptr),
      as_stream(stream_ptr));
}

void mlx_fast_softmax(
    void* result_ptr,
    const void* a_ptr,
    const int* axes,
    size_t axes_num,
    bool precise,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::softmax(
      as_const_array(a_ptr),
      std::vector<int>(axes, axes + axes_num),
      precise,
      as_stream(stream_ptr));
}

// ============================================================================
// Element-wise binary ops
// ============================================================================

void mlx_fast_subtract(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::subtract(
      as_const_array(a_ptr),
      as_const_array(b_ptr),
      as_stream(stream_ptr));
}

void mlx_fast_divide(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::divide(
      as_const_array(a_ptr),
      as_const_array(b_ptr),
      as_stream(stream_ptr));
}

void mlx_fast_maximum(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::maximum(
      as_const_array(a_ptr),
      as_const_array(b_ptr),
      as_stream(stream_ptr));
}

void mlx_fast_minimum(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::minimum(
      as_const_array(a_ptr),
      as_const_array(b_ptr),
      as_stream(stream_ptr));
}

void mlx_fast_power(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::power(
      as_const_array(a_ptr),
      as_const_array(b_ptr),
      as_stream(stream_ptr));
}

void mlx_fast_where(
    void* result_ptr,
    const void* condition_ptr,
    const void* x_ptr,
    const void* y_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::where(
      as_const_array(condition_ptr),
      as_const_array(x_ptr),
      as_const_array(y_ptr),
      as_stream(stream_ptr));
}

// ============================================================================
// Element-wise unary ops
// ============================================================================

void mlx_fast_negative(
    void* result_ptr,
    const void* a_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::negative(
      as_const_array(a_ptr),
      as_stream(stream_ptr));
}

void mlx_fast_abs(
    void* result_ptr,
    const void* a_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::abs(
      as_const_array(a_ptr),
      as_stream(stream_ptr));
}

void mlx_fast_exp(
    void* result_ptr,
    const void* a_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::exp(
      as_const_array(a_ptr),
      as_stream(stream_ptr));
}

void mlx_fast_log(
    void* result_ptr,
    const void* a_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::log(
      as_const_array(a_ptr),
      as_stream(stream_ptr));
}

void mlx_fast_sigmoid(
    void* result_ptr,
    const void* a_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::sigmoid(
      as_const_array(a_ptr),
      as_stream(stream_ptr));
}

void mlx_fast_logical_not(
    void* result_ptr,
    const void* a_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::logical_not(
      as_const_array(a_ptr),
      as_stream(stream_ptr));
}

// silu = x * sigmoid(x), composed from primitives (no C++ silu in MLX)
void mlx_fast_silu(
    void* result_ptr,
    const void* a_ptr,
    const void* stream_ptr) {
  const auto& a = as_const_array(a_ptr);
  const auto& s = as_stream(stream_ptr);
  as_array(result_ptr) = mlx::core::multiply(
      a, mlx::core::sigmoid(a, s), s);
}

// log_softmax = x - log(sum(exp(x)))
// Using softmax then log is numerically equivalent: log(softmax(x))
// But the dedicated approach is: x - logsumexp(x)
void mlx_fast_log_softmax(
    void* result_ptr,
    const void* a_ptr,
    const int* axes,
    size_t axes_num,
    const void* stream_ptr) {
  const auto& a = as_const_array(a_ptr);
  const auto& s = as_stream(stream_ptr);
  auto ax = std::vector<int>(axes, axes + axes_num);
  auto lse = mlx::core::logsumexp(a, ax, true, s);
  as_array(result_ptr) = mlx::core::subtract(a, lse, s);
}

// ============================================================================
// Comparison ops
// ============================================================================

void mlx_fast_equal(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::equal(
      as_const_array(a_ptr),
      as_const_array(b_ptr),
      as_stream(stream_ptr));
}

void mlx_fast_less(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::less(
      as_const_array(a_ptr),
      as_const_array(b_ptr),
      as_stream(stream_ptr));
}

void mlx_fast_greater(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::greater(
      as_const_array(a_ptr),
      as_const_array(b_ptr),
      as_stream(stream_ptr));
}

void mlx_fast_greater_equal(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::greater_equal(
      as_const_array(a_ptr),
      as_const_array(b_ptr),
      as_stream(stream_ptr));
}

void mlx_fast_less_equal(
    void* result_ptr,
    const void* a_ptr,
    const void* b_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::less_equal(
      as_const_array(a_ptr),
      as_const_array(b_ptr),
      as_stream(stream_ptr));
}

// ============================================================================
// Reduction ops
// ============================================================================

void mlx_fast_sum_axis(
    void* result_ptr,
    const void* a_ptr,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::sum(
      as_const_array(a_ptr),
      std::vector<int>(axes, axes + axes_num),
      keepdims,
      as_stream(stream_ptr));
}

void mlx_fast_sum_all(
    void* result_ptr,
    const void* a_ptr,
    bool keepdims,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::sum(
      as_const_array(a_ptr),
      keepdims,
      as_stream(stream_ptr));
}

void mlx_fast_mean_axis(
    void* result_ptr,
    const void* a_ptr,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::mean(
      as_const_array(a_ptr),
      std::vector<int>(axes, axes + axes_num),
      keepdims,
      as_stream(stream_ptr));
}

void mlx_fast_mean_all(
    void* result_ptr,
    const void* a_ptr,
    bool keepdims,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::mean(
      as_const_array(a_ptr),
      keepdims,
      as_stream(stream_ptr));
}

void mlx_fast_max_axis(
    void* result_ptr,
    const void* a_ptr,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::max(
      as_const_array(a_ptr),
      std::vector<int>(axes, axes + axes_num),
      keepdims,
      as_stream(stream_ptr));
}

void mlx_fast_max_all(
    void* result_ptr,
    const void* a_ptr,
    bool keepdims,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::max(
      as_const_array(a_ptr),
      keepdims,
      as_stream(stream_ptr));
}

void mlx_fast_min_axis(
    void* result_ptr,
    const void* a_ptr,
    const int* axes,
    size_t axes_num,
    bool keepdims,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::min(
      as_const_array(a_ptr),
      std::vector<int>(axes, axes + axes_num),
      keepdims,
      as_stream(stream_ptr));
}

void mlx_fast_min_all(
    void* result_ptr,
    const void* a_ptr,
    bool keepdims,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::min(
      as_const_array(a_ptr),
      keepdims,
      as_stream(stream_ptr));
}

void mlx_fast_argmax_axis(
    void* result_ptr,
    const void* a_ptr,
    int axis,
    bool keepdims,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::argmax(
      as_const_array(a_ptr),
      axis,
      keepdims,
      as_stream(stream_ptr));
}

void mlx_fast_argmax_all(
    void* result_ptr,
    const void* a_ptr,
    bool keepdims,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::argmax(
      as_const_array(a_ptr),
      keepdims,
      as_stream(stream_ptr));
}

void mlx_fast_argmin_axis(
    void* result_ptr,
    const void* a_ptr,
    int axis,
    bool keepdims,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::argmin(
      as_const_array(a_ptr),
      axis,
      keepdims,
      as_stream(stream_ptr));
}

void mlx_fast_argmin_all(
    void* result_ptr,
    const void* a_ptr,
    bool keepdims,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::argmin(
      as_const_array(a_ptr),
      keepdims,
      as_stream(stream_ptr));
}

// ============================================================================
// Shape / indexing ops
// ============================================================================

void mlx_fast_concatenate(
    void* result_ptr,
    const void* const* arrays_ptr,
    size_t num_arrays,
    int axis,
    const void* stream_ptr) {
  std::vector<mlx::core::array> arrays;
  arrays.reserve(num_arrays);
  for (size_t i = 0; i < num_arrays; i++) {
    arrays.push_back(as_const_array(arrays_ptr[i]));
  }
  as_array(result_ptr) = mlx::core::concatenate(
      std::move(arrays), axis, as_stream(stream_ptr));
}

void mlx_fast_split_equal(
    void** results_ptr,
    size_t* num_results,
    const void* a_ptr,
    int num_splits,
    int axis,
    const void* stream_ptr) {
  auto parts = mlx::core::split(
      as_const_array(a_ptr), num_splits, axis, as_stream(stream_ptr));
  *num_results = parts.size();
  for (size_t i = 0; i < parts.size(); i++) {
    as_array(results_ptr[i]) = std::move(parts[i]);
  }
}

void mlx_fast_split_indices(
    void** results_ptr,
    size_t* num_results,
    const void* a_ptr,
    const int32_t* indices,
    size_t indices_num,
    int axis,
    const void* stream_ptr) {
  auto parts = mlx::core::split(
      as_const_array(a_ptr),
      mlx::core::Shape(indices, indices + indices_num),
      axis,
      as_stream(stream_ptr));
  *num_results = parts.size();
  for (size_t i = 0; i < parts.size(); i++) {
    as_array(results_ptr[i]) = std::move(parts[i]);
  }
}

void mlx_fast_expand_dims(
    void* result_ptr,
    const void* a_ptr,
    const int* axes,
    size_t axes_num,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::expand_dims(
      as_const_array(a_ptr),
      std::vector<int>(axes, axes + axes_num),
      as_stream(stream_ptr));
}

void mlx_fast_expand_dims_axis(
    void* result_ptr,
    const void* a_ptr,
    int axis,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::expand_dims(
      as_const_array(a_ptr),
      axis,
      as_stream(stream_ptr));
}

void mlx_fast_squeeze_axes(
    void* result_ptr,
    const void* a_ptr,
    const int* axes,
    size_t axes_num,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::squeeze(
      as_const_array(a_ptr),
      std::vector<int>(axes, axes + axes_num),
      as_stream(stream_ptr));
}

void mlx_fast_squeeze_axis(
    void* result_ptr,
    const void* a_ptr,
    int axis,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::squeeze(
      as_const_array(a_ptr),
      axis,
      as_stream(stream_ptr));
}

void mlx_fast_squeeze_all(
    void* result_ptr,
    const void* a_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::squeeze(
      as_const_array(a_ptr),
      as_stream(stream_ptr));
}

void mlx_fast_astype(
    void* result_ptr,
    const void* a_ptr,
    uint8_t dtype_val,
    uint8_t dtype_size,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::astype(
      as_const_array(a_ptr),
      as_dtype(dtype_val, dtype_size),
      as_stream(stream_ptr));
}

void mlx_fast_flatten(
    void* result_ptr,
    const void* a_ptr,
    int start_axis,
    int end_axis,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::flatten(
      as_const_array(a_ptr),
      start_axis,
      end_axis,
      as_stream(stream_ptr));
}

void mlx_fast_flatten_all(
    void* result_ptr,
    const void* a_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::flatten(
      as_const_array(a_ptr),
      as_stream(stream_ptr));
}

void mlx_fast_take_axis(
    void* result_ptr,
    const void* a_ptr,
    const void* indices_ptr,
    int axis,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::take(
      as_const_array(a_ptr),
      as_const_array(indices_ptr),
      axis,
      as_stream(stream_ptr));
}

void mlx_fast_take(
    void* result_ptr,
    const void* a_ptr,
    const void* indices_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::take(
      as_const_array(a_ptr),
      as_const_array(indices_ptr),
      as_stream(stream_ptr));
}

void mlx_fast_take_along_axis(
    void* result_ptr,
    const void* a_ptr,
    const void* indices_ptr,
    int axis,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::take_along_axis(
      as_const_array(a_ptr),
      as_const_array(indices_ptr),
      axis,
      as_stream(stream_ptr));
}

void mlx_fast_put_along_axis(
    void* result_ptr,
    const void* a_ptr,
    const void* indices_ptr,
    const void* values_ptr,
    int axis,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::put_along_axis(
      as_const_array(a_ptr),
      as_const_array(indices_ptr),
      as_const_array(values_ptr),
      axis,
      as_stream(stream_ptr));
}

void mlx_fast_gather(
    void* result_ptr,
    const void* a_ptr,
    const void* indices_ptr,
    int axis,
    const int32_t* slice_sizes,
    size_t slice_sizes_num,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::gather(
      as_const_array(a_ptr),
      as_const_array(indices_ptr),
      axis,
      mlx::core::Shape(slice_sizes, slice_sizes + slice_sizes_num),
      as_stream(stream_ptr));
}

// ============================================================================
// Sort / partition ops
// ============================================================================

void mlx_fast_sort_axis(
    void* result_ptr,
    const void* a_ptr,
    int axis,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::sort(
      as_const_array(a_ptr),
      axis,
      as_stream(stream_ptr));
}

void mlx_fast_sort_all(
    void* result_ptr,
    const void* a_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::sort(
      as_const_array(a_ptr),
      as_stream(stream_ptr));
}

void mlx_fast_argsort_axis(
    void* result_ptr,
    const void* a_ptr,
    int axis,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::argsort(
      as_const_array(a_ptr),
      axis,
      as_stream(stream_ptr));
}

void mlx_fast_argsort_all(
    void* result_ptr,
    const void* a_ptr,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::argsort(
      as_const_array(a_ptr),
      as_stream(stream_ptr));
}

void mlx_fast_argpartition_axis(
    void* result_ptr,
    const void* a_ptr,
    int kth,
    int axis,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::argpartition(
      as_const_array(a_ptr),
      kth,
      axis,
      as_stream(stream_ptr));
}

void mlx_fast_argpartition_all(
    void* result_ptr,
    const void* a_ptr,
    int kth,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::argpartition(
      as_const_array(a_ptr),
      kth,
      as_stream(stream_ptr));
}

// ============================================================================
// Cumulative ops
// ============================================================================

void mlx_fast_cumsum_axis(
    void* result_ptr,
    const void* a_ptr,
    int axis,
    bool reverse,
    bool inclusive,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::cumsum(
      as_const_array(a_ptr),
      axis,
      reverse,
      inclusive,
      as_stream(stream_ptr));
}

void mlx_fast_cumsum_all(
    void* result_ptr,
    const void* a_ptr,
    bool reverse,
    bool inclusive,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::cumsum(
      as_const_array(a_ptr),
      reverse,
      inclusive,
      as_stream(stream_ptr));
}

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
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::conv1d(
      as_const_array(input_ptr),
      as_const_array(weight_ptr),
      stride,
      padding,
      dilation,
      groups,
      as_stream(stream_ptr));
}

// ============================================================================
// Array creation
// ============================================================================

void mlx_fast_zeros(
    void* result_ptr,
    const int32_t* shape,
    size_t shape_num,
    uint8_t dtype_val,
    uint8_t dtype_size,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::zeros(
      mlx::core::Shape(shape, shape + shape_num),
      as_dtype(dtype_val, dtype_size),
      as_stream(stream_ptr));
}

void mlx_fast_ones(
    void* result_ptr,
    const int32_t* shape,
    size_t shape_num,
    uint8_t dtype_val,
    uint8_t dtype_size,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::ones(
      mlx::core::Shape(shape, shape + shape_num),
      as_dtype(dtype_val, dtype_size),
      as_stream(stream_ptr));
}

void mlx_fast_full_float(
    void* result_ptr,
    const int32_t* shape,
    size_t shape_num,
    float val,
    uint8_t dtype_val,
    uint8_t dtype_size,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::full(
      mlx::core::Shape(shape, shape + shape_num),
      val,
      as_dtype(dtype_val, dtype_size),
      as_stream(stream_ptr));
}

void mlx_fast_arange_float(
    void* result_ptr,
    double start,
    double stop,
    double step,
    uint8_t dtype_val,
    uint8_t dtype_size,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::arange(
      start, stop, step,
      as_dtype(dtype_val, dtype_size),
      as_stream(stream_ptr));
}

void mlx_fast_arange_int(
    void* result_ptr,
    int start,
    int stop,
    int step,
    const void* stream_ptr) {
  as_array(result_ptr) = mlx::core::arange(
      start, stop, step,
      as_stream(stream_ptr));
}

// ============================================================================
// Utility
// ============================================================================

void* mlx_fast_alloc_array(void) {
  return new mlx::core::array(0.0f);
}

void mlx_fast_free_array(void* arr_ptr) {
  delete static_cast<mlx::core::array*>(arr_ptr);
}

const void* mlx_fast_default_gpu_stream(void) {
  static mlx::core::Stream s = mlx::core::default_stream(
      mlx::core::Device(mlx::core::Device::gpu));
  return &s;
}

const void* mlx_fast_default_cpu_stream(void) {
  static mlx::core::Stream s = mlx::core::default_stream(
      mlx::core::Device(mlx::core::Device::cpu));
  return &s;
}

} // extern "C"
