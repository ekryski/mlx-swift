// Warp Decode MoE: Fused Down+WeightedSum kernel
//
// Each threadgroup computes 8 output neurons of the FINAL MoE output by
// looping over all topK experts and folding routing weights into the
// accumulator. Eliminates the separate weighted-sum dispatch.
//
// Grid: (1, ceil(outputDims / 8), 1)
//   tid.y: output neuron tile (8 rows per threadgroup)
//
// For each output row i:
//   result = 0
//   for k in 0..<topK:
//     expert_id = indices[k]
//     dot_val = dot(activated[k, :], down_W[expert_id, i, :])
//     result += scores[k] * dot_val     ← routing weight folded in!
//   out[i] = result
//
// Replaces: gatherQuantizedMM(down) + weighted_sum(scores)
// Saves: 1 dispatch + eliminates per-expert output buffers

#include <metal_common>
#include <metal_simdgroup>

#include "utils.h"

using namespace metal;

template <int values_per_thread>
inline float qdot_4bit(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    accum +=
        (x_thread[4 * i]     * float(ws[i] & 0x000f) +
         x_thread[4 * i + 1] * float(ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * float(ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * float(ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}

template <int values_per_thread>
inline float qdot_4bit_safe(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum,
    int remaining) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    int base = 4 * i;
    if (base < remaining) accum += x_thread[base] * float(ws[i] & 0x000f);
    if (base + 1 < remaining) accum += x_thread[base + 1] * float(ws[i] & 0x00f0);
    if (base + 2 < remaining) accum += x_thread[base + 2] * float(ws[i] & 0x0f00);
    if (base + 3 < remaining) accum += x_thread[base + 3] * float(ws[i] & 0xf000);
  }
  return scale * accum + sum * bias;
}

// ─── Main kernel ────────────────────────────────────────────────────────────

template <typename T, int group_size>
[[kernel]] void warp_moe_down(
    // Activated intermediates: [topK, hiddenDims] (from gate_up kernel)
    const device T* activated [[buffer(0)]],
    // Down projection weights: [numExperts, outputDims, hiddenDims_packed]
    const device uint32_t* w [[buffer(1)]],
    const device T* scales [[buffer(2)]],
    const device T* biases [[buffer(3)]],
    // Expert indices: [topK]
    const device int32_t* indices [[buffer(4)]],
    // Routing scores: [topK] (softmax weights)
    const device T* scores [[buffer(5)]],
    // Output: [outputDims] — final MoE output (weighted sum folded in)
    device T* out [[buffer(6)]],
    // Dimensions
    constant int& hidden_dims [[buffer(7)]],    // input to down proj (= hiddenDims)
    constant int& out_vec_size [[buffer(8)]],   // output of down proj (= inputDims)
    constant int& top_k [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  constexpr int SIMD_SIZE = 32;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = 8;
  constexpr int values_per_thread = pack_factor;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int bytes_per_pack = 4;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) return;

  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  const int in_vec_size = hidden_dims;  // down proj input = hiddenDims
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;

  constexpr float qdot_prescale[8] = {
      1.0f, 1.0f/16.0f, 1.0f/256.0f, 1.0f/4096.0f,
      1.0f, 1.0f/16.0f, 1.0f/256.0f, 1.0f/4096.0f
  };

  // Accumulate weighted results across all topK experts
  thread float result[results_per_simdgroup] = {0};

  // Shared memory for activated vector (reloaded per expert)
  threadgroup T shared_act[8192];  // max hiddenDims across models

  uint total_threads = num_simdgroups * SIMD_SIZE;
  uint thread_id = simd_gid * SIMD_SIZE + simd_lid;

  // Loop over topK experts
  for (int ek = 0; ek < top_k; ek++) {
    int expert_id = indices[ek];
    float score = float(scores[ek]);

    // Load this expert's activated vector to shared memory
    const device T* act_k = activated + ek * hidden_dims;
    for (uint i = thread_id; i < uint(in_vec_size); i += total_threads) {
      shared_act[i] = act_k[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute down projection: dot(activated_k, down_W[expert_id, row, :])
    int expert_w_offset = expert_id * out_vec_size * in_vec_size_w;
    int expert_s_offset = expert_id * out_vec_size * in_vec_size_g;

    const device uint8_t* ws = (const device uint8_t*)w + expert_w_offset
        + used_out_row * in_vec_size_w + simd_lid * bytes_per_pack;
    const device T* sc = scales + expert_s_offset
        + used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    const device T* bi = biases + expert_s_offset
        + used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

    uint x_offset = simd_lid * values_per_thread;
    thread float x_thread[values_per_thread];
    thread float expert_result[results_per_simdgroup] = {0};

    const device uint8_t* ws_iter = ws;
    const device T* sc_iter = sc;
    const device T* bi_iter = bi;
    uint x_off = x_offset;
    int kk = 0;

    for (; kk < in_vec_size - block_size; kk += block_size) {
      float sum = 0;
      for (int i = 0; i < values_per_thread; i++) {
        float val = float(shared_act[x_off + i]);
        sum += val;
        x_thread[i] = val * qdot_prescale[i];
      }

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws_iter + row * in_vec_size_w);
        float s = float(sc_iter[row * in_vec_size_g]);
        float b = float(bi_iter[row * in_vec_size_g]);
        expert_result[row] += qdot_4bit<values_per_thread>(wl, x_thread, s, b, sum);
      }

      ws_iter += block_size * bytes_per_pack / pack_factor;
      sc_iter += block_size / group_size;
      bi_iter += block_size / group_size;
      x_off += block_size;
    }

    // Remaining elements
    const int remaining = clamp(
        int(in_vec_size) - kk - int(simd_lid * values_per_thread),
        0, values_per_thread);
    if (remaining > 0) {
      float sum = 0;
      for (int i = 0; i < values_per_thread; i++) {
        if (i < remaining) {
          float val = float(shared_act[x_off + i]);
          sum += val;
          x_thread[i] = val * qdot_prescale[i];
        } else {
          x_thread[i] = 0;
        }
      }

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws_iter + row * in_vec_size_w);
        float s = float(sc_iter[row * in_vec_size_g]);
        float b = float(bi_iter[row * in_vec_size_g]);
        expert_result[row] += qdot_4bit_safe<values_per_thread>(
            wl, x_thread, s, b, sum, remaining);
      }
    }

    // SIMD reduce per-expert results and fold routing weight
    for (int row = 0; row < results_per_simdgroup; row++) {
      float dot_val = simd_sum(expert_result[row]);
      // Fold routing weight: only lane 0 has the reduced value
      if (simd_lid == 0) {
        result[row] += score * dot_val;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write final weighted output
  for (int row = 0; row < results_per_simdgroup; row++) {
    if (simd_lid == 0 && used_out_row + row < out_vec_size) {
      out[used_out_row + row] = static_cast<T>(result[row]);
    }
  }
}

// ─── Instantiation ──────────────────────────────────────────────────────────

#define instantiate_warp_moe_down(type, tname, gs) \
  template [[host_name("warp_moe_down_" #tname "_gs" #gs)]] \
  [[kernel]] void warp_moe_down<type, gs>( \
    const device type*, const device uint32_t*, const device type*, \
    const device type*, const device int32_t*, const device type*, \
    device type*, \
    constant int&, constant int&, constant int&, \
    uint3, uint, uint);

instantiate_warp_moe_down(half, float16, 64)
instantiate_warp_moe_down(bfloat16_t, bfloat16, 64)
instantiate_warp_moe_down(half, float16, 128)
instantiate_warp_moe_down(bfloat16_t, bfloat16, 128)
