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

extern "C" {

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
