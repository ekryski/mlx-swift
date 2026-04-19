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

/**
 * Reset the cumulative dispatch counter on the default Metal command
 * encoder. Call before a region of interest, then `mlx_metal_total_dispatches`
 * after, to count the Metal kernel launches MLX issued for that region.
 * Useful for auditing per-token dispatch counts (e.g. for ICB feasibility
 * studies).
 */
int mlx_metal_reset_dispatch_counter(void);

/** Read the cumulative dispatch counter since the last reset. */
int mlx_metal_total_dispatches(uint64_t* res);

/**
 * Start recording Metal kernel pipeline-state labels on every dispatch.
 * Calling this clears any prior log. Enables the ICB dispatch-list
 * stability audit.
 */
int mlx_metal_start_kernel_log(void);

/** Stop recording kernel labels. */
int mlx_metal_stop_kernel_log(void);

/** Number of entries in the current kernel log. */
int mlx_metal_kernel_log_size(size_t* res);

/**
 * Write the kernel label at index i into `*label_out`. Returns nullptr
 * in `*label_out` if i is out of range. The C string is valid until the
 * next `mlx_metal_start_kernel_log()`.
 */
int mlx_metal_kernel_log_at(size_t i, const char** label_out);

/**
 * Indirect Command Buffer (ICB) capture & replay.
 *
 * The ICB path lets callers pre-encode a stable sequence of compute
 * dispatches once and replay it per decode step, trading one-time build
 * cost for substantially lower per-replay CPU encoding overhead. See
 * mlx/backend/metal/icb.h for semantics, invariants, and the
 * barrier-splitting design.
 */

/** Opaque handle to an IndirectCommandRecorder. */
typedef struct mlx_metal_icb_recorder_ {
  void* ctx;
} mlx_metal_icb_recorder;

/**
 * Whether the current GPU supports MTLIndirectCommandBuffer-backed
 * compute. Effectively true on every M-series device.
 */
int mlx_metal_icb_is_supported(bool* res);

/**
 * Begin recording on `stream`'s CommandEncoder. Subsequent dispatches
 * emitted through the encoder accumulate into an ICB recorder instead
 * of running live. `max_commands_per_segment` caps the size of each ICB
 * segment (a segment is started per memory barrier); `bytes_arena_cap`
 * is the shared pool for spilled `setBytes` payloads.
 */
int mlx_metal_icb_begin_recording(
    mlx_stream stream,
    size_t max_commands_per_segment,
    size_t bytes_arena_cap);

/**
 * Finalize recording on `stream` and move the recorder out via `*out`.
 * The caller owns `*out` and must release it with
 * `mlx_metal_icb_recorder_free`.
 */
int mlx_metal_icb_end_recording(
    mlx_stream stream,
    mlx_metal_icb_recorder* out);

/**
 * Cancel the current recording session without producing a recorder.
 * Safe to call whether or not the stream is currently recording — used
 * to clean up after an exception in the recording block.
 */
int mlx_metal_icb_abort_recording(mlx_stream stream);

/**
 * Replay a previously-captured recording on `stream`. The live encoder
 * issues `useResource` for every buffer referenced by each segment and
 * `executeCommandsInBuffer` for the segment's range, with a memory
 * barrier between consecutive segments.
 */
int mlx_metal_icb_replay(mlx_stream stream, mlx_metal_icb_recorder rec);

/**
 * Associate `name_id` with the MTL::Buffer underlying `array` during
 * the active recording on `stream`. Any already-recorded dispatch
 * that bound that buffer is tagged immediately; any subsequent bind
 * of the same buffer under this recorder is also tagged on end_command.
 * Returns an error if `stream` is not currently recording.
 */
int mlx_metal_icb_tag_binding(
    mlx_stream stream,
    uint32_t name_id,
    const mlx_array array);

/**
 * Replay `rec` on `stream` with per-name buffer overrides. For each
 * `i` in `[0, n_overrides)`, every dispatch previously tagged with
 * `names[i]` has its kernel-buffer binding rewritten to the buffer
 * underlying `arrays[i]` (offset included from the array's storage)
 * before the ICB is executed. Tags not present in `names` keep their
 * recorded binding; names not present in the recorder's tag table
 * are silently skipped. Override writes mutate the recorder's ICBs
 * in place — see `IndirectCommandRecorder::replay_with_overrides`.
 */
int mlx_metal_icb_replay_with_overrides(
    mlx_stream stream,
    mlx_metal_icb_recorder rec,
    const uint32_t* names,
    const mlx_array* arrays,
    size_t n_overrides);

/**
 * Number of ICB segments (barrier-separated blocks) in the recording.
 */
int mlx_metal_icb_recorder_num_segments(
    mlx_metal_icb_recorder rec,
    size_t* res);

/**
 * Total number of commands across all segments.
 */
int mlx_metal_icb_recorder_size(mlx_metal_icb_recorder rec, size_t* res);

/**
 * Release a recorder. Safe to pass a recorder whose `ctx` is NULL.
 */
int mlx_metal_icb_recorder_free(mlx_metal_icb_recorder rec);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
