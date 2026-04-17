/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_METAL_H
#define MLX_METAL_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "mlx/c/array.h"
#include "mlx/c/closure.h"
#include "mlx/c/distributed_group.h"
#include "mlx/c/io_types.h"
#include "mlx/c/map.h"
#include "mlx/c/stream.h"
#include "mlx/c/string.h"
#include "mlx/c/vector.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup metal Metal specific operations
 */
/**@{*/

int mlx_metal_is_available(bool* res);
int mlx_metal_start_capture(const char* path);
int mlx_metal_stop_capture(void);
int mlx_metal_reset_dispatch_counter(void);
int mlx_metal_total_dispatches(uint64_t* res);
int mlx_metal_start_kernel_log(void);
int mlx_metal_stop_kernel_log(void);
int mlx_metal_kernel_log_size(size_t* res);
int mlx_metal_kernel_log_at(size_t i, const char** label_out);

typedef struct mlx_metal_icb_recorder_ {
  void* ctx;
} mlx_metal_icb_recorder;

int mlx_metal_icb_is_supported(bool* res);
int mlx_metal_icb_begin_recording(
    mlx_stream stream,
    size_t max_commands_per_segment,
    size_t bytes_arena_cap);
int mlx_metal_icb_end_recording(
    mlx_stream stream,
    mlx_metal_icb_recorder* out);
int mlx_metal_icb_abort_recording(mlx_stream stream);
int mlx_metal_icb_replay(mlx_stream stream, mlx_metal_icb_recorder rec);
int mlx_metal_icb_recorder_num_segments(
    mlx_metal_icb_recorder rec,
    size_t* res);
int mlx_metal_icb_recorder_size(mlx_metal_icb_recorder rec, size_t* res);
int mlx_metal_icb_recorder_free(mlx_metal_icb_recorder rec);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
