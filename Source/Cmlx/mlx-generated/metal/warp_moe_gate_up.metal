// Warp Decode MoE: Fused Gate+Up+Activation kernel
//
// Each threadgroup computes 8 output neurons of the activated intermediate
// for ONE expert (selected by tid.z). Input x is loaded to shared memory
// once and reused across all output rows.
//
// Grid: (1, ceil(hiddenDims / 8), topK)
//   tid.y: output neuron tile (8 rows per threadgroup)
//   tid.z: expert index within topK selection
//
// For each output row i:
//   gate_val = dot(x, gate_up_W[expert_id, i, :])       — first half of fused weight
//   up_val   = dot(x, gate_up_W[expert_id, i + H, :])   — second half
//   out[i]   = activation(gate_val) * up_val
//
// Replaces: gatherQuantizedMM(gate_up) + split + activation + multiply
// Saves: 1 dispatch + eliminates intermediate gateUp tensor materialization

#include <metal_common>
#include <metal_simdgroup>
#include <metal_math>

#include "utils.h"

using namespace metal;

// ─── 4-bit quantized dot product (inlined from rms_norm_qgemv.metal) ────────

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

// ─── Activation functions ───────────────────────────────────────────────────

inline float silu(float x) {
  return x / (1.0f + metal::precise::exp(-x));
}

inline float gelu_approx(float x) {
  // tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  constexpr float kSqrt2OverPi = 0.7978845608f;
  float x3 = x * x * x;
  float inner = kSqrt2OverPi * (x + 0.044715f * x3);
  return 0.5f * x * (1.0f + metal::precise::tanh(inner));
}

// SwiGLU for GPT-OSS: gate = clip(gate, max=7), up = clip(up, -7, 7),
// sig = sigmoid(1.702 * gate), output = gate * sig * (up + 1.0)
inline float swiglu_gate(float gate, float up) {
  gate = metal::clamp(gate, -7.0f, 7.0f);
  up = metal::clamp(up, -7.0f, 7.0f);
  float sig = 1.0f / (1.0f + metal::precise::exp(-1.702f * gate));
  return gate * sig * (up + 1.0f);
}

// ─── Main kernel ────────────────────────────────────────────────────────────

// activation_type: 0=silu, 1=gelu_approx, 2=swiglu
template <typename T, int group_size, int activation_type>
[[kernel]] void warp_moe_gate_up(
    const device T* x [[buffer(0)]],
    // gate_up weights: [numExperts, 2*hiddenDims, inputDims_packed]
    const device uint32_t* w [[buffer(1)]],
    const device T* scales [[buffer(2)]],
    const device T* biases [[buffer(3)]],
    // Expert indices: [topK]
    const device int32_t* indices [[buffer(4)]],
    // Output: [topK, hiddenDims]
    device T* out [[buffer(5)]],
    // Dimensions
    constant int& in_vec_size [[buffer(6)]],    // inputDims
    constant int& hidden_dims [[buffer(7)]],    // hiddenDims (half of gate_up output)
    constant int& top_k [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  constexpr int SIMD_SIZE = 32;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = 8;
  constexpr int values_per_thread = pack_factor;
  constexpr int block_size = values_per_thread * SIMD_SIZE;  // 256
  constexpr int bytes_per_pack = 4;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const int expert_k = tid.z;  // which of the topK experts this threadgroup handles
  if (expert_k >= top_k) return;

  const int expert_id = indices[expert_k];
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= hidden_dims) return;

  // ── Phase 1: Load x to shared memory ──────────────────────────────────
  threadgroup T shared_x[4096];
  uint total_threads = num_simdgroups * SIMD_SIZE;
  uint thread_id = simd_gid * SIMD_SIZE + simd_lid;
  for (uint i = thread_id; i < uint(in_vec_size); i += total_threads) {
    shared_x[i] = x[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ── Phase 2: Compute gate + up dot products for each output row ───────

  const int gate_up_out_dim = 2 * hidden_dims;
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;  // packed row stride
  const int in_vec_size_g = in_vec_size / group_size;  // scales/biases row stride
  // Weight stride per expert: expert_id * gate_up_out_dim rows
  const int expert_w_offset = expert_id * gate_up_out_dim * in_vec_size_w;
  const int expert_s_offset = expert_id * gate_up_out_dim * in_vec_size_g;

  const int used_out_row = min(hidden_dims - results_per_simdgroup, out_row);

  constexpr float qdot_prescale[8] = {
      1.0f, 1.0f/16.0f, 1.0f/256.0f, 1.0f/4096.0f,
      1.0f, 1.0f/16.0f, 1.0f/256.0f, 1.0f/4096.0f
  };

  thread float gate_result[results_per_simdgroup] = {0};
  thread float up_result[results_per_simdgroup] = {0};

  // Pointers to gate rows (first H rows) and up rows (next H rows) for this expert
  const device uint8_t* gate_ws = (const device uint8_t*)w + expert_w_offset
      + used_out_row * in_vec_size_w + simd_lid * bytes_per_pack;
  const device T* gate_sc = scales + expert_s_offset
      + used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* gate_bi = biases + expert_s_offset
      + used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

  const device uint8_t* up_ws = (const device uint8_t*)w + expert_w_offset
      + (used_out_row + hidden_dims) * in_vec_size_w + simd_lid * bytes_per_pack;
  const device T* up_sc = scales + expert_s_offset
      + (used_out_row + hidden_dims) * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* up_bi = biases + expert_s_offset
      + (used_out_row + hidden_dims) * in_vec_size_g + simd_lid / scale_step_per_thread;

  uint x_offset = simd_lid * values_per_thread;
  thread float x_thread[values_per_thread];
  int k = 0;

  for (; k < in_vec_size - block_size; k += block_size) {
    float sum = 0;
    for (int i = 0; i < values_per_thread; i++) {
      float val = float(shared_x[x_offset + i]);
      sum += val;
      x_thread[i] = val * qdot_prescale[i];
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      // Gate dot product
      auto gwl = (const device uint8_t*)(gate_ws + row * in_vec_size_w);
      float gs = float(gate_sc[row * in_vec_size_g]);
      float gb = float(gate_bi[row * in_vec_size_g]);
      gate_result[row] += qdot_4bit<values_per_thread>(gwl, x_thread, gs, gb, sum);

      // Up dot product
      auto uwl = (const device uint8_t*)(up_ws + row * in_vec_size_w);
      float us = float(up_sc[row * in_vec_size_g]);
      float ub = float(up_bi[row * in_vec_size_g]);
      up_result[row] += qdot_4bit<values_per_thread>(uwl, x_thread, us, ub, sum);
    }

    gate_ws += block_size * bytes_per_pack / pack_factor;
    gate_sc += block_size / group_size;
    gate_bi += block_size / group_size;
    up_ws += block_size * bytes_per_pack / pack_factor;
    up_sc += block_size / group_size;
    up_bi += block_size / group_size;
    x_offset += block_size;
  }

  // Handle remaining elements
  const int remaining = clamp(
      int(in_vec_size) - k - int(simd_lid * values_per_thread),
      0, values_per_thread);
  if (remaining > 0) {
    float sum = 0;
    for (int i = 0; i < values_per_thread; i++) {
      if (i < remaining) {
        float val = float(shared_x[x_offset + i]);
        sum += val;
        x_thread[i] = val * qdot_prescale[i];
      } else {
        x_thread[i] = 0;
      }
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto gwl = (const device uint8_t*)(gate_ws + row * in_vec_size_w);
      float gs = float(gate_sc[row * in_vec_size_g]);
      float gb = float(gate_bi[row * in_vec_size_g]);
      gate_result[row] += qdot_4bit_safe<values_per_thread>(gwl, x_thread, gs, gb, sum, remaining);

      auto uwl = (const device uint8_t*)(up_ws + row * in_vec_size_w);
      float us = float(up_sc[row * in_vec_size_g]);
      float ub = float(up_bi[row * in_vec_size_g]);
      up_result[row] += qdot_4bit_safe<values_per_thread>(uwl, x_thread, us, ub, sum, remaining);
    }
  }

  // ── Phase 3: SIMD reduction + activation + write ──────────────────────
  device T* expert_out = out + expert_k * hidden_dims + used_out_row;

  for (int row = 0; row < results_per_simdgroup; row++) {
    float g = simd_sum(gate_result[row]);
    float u = simd_sum(up_result[row]);

    if (simd_lid == 0 && used_out_row + row < hidden_dims) {
      float activated;
      if (activation_type == 0) {
        activated = silu(g) * u;
      } else if (activation_type == 1) {
        activated = gelu_approx(g) * u;
      } else {
        activated = swiglu_gate(g, u);
      }
      expert_out[row] = static_cast<T>(activated);
    }
  }
}

// ─── Instantiation ──────────────────────────────────────────────────────────

#define instantiate_warp_moe_gate_up(type, tname, gs, act) \
  template [[host_name("warp_moe_gate_up_" #tname "_gs" #gs "_act" #act)]] \
  [[kernel]] void warp_moe_gate_up<type, gs, act>( \
    const device type*, const device uint32_t*, const device type*, \
    const device type*, const device int32_t*, device type*, \
    constant int&, constant int&, constant int&, \
    uint3, uint, uint);

// silu (Qwen3.5)
instantiate_warp_moe_gate_up(half, float16, 64, 0)
instantiate_warp_moe_gate_up(bfloat16_t, bfloat16, 64, 0)
instantiate_warp_moe_gate_up(half, float16, 128, 0)
instantiate_warp_moe_gate_up(bfloat16_t, bfloat16, 128, 0)

// gelu_approx (Gemma4)
instantiate_warp_moe_gate_up(half, float16, 64, 1)
instantiate_warp_moe_gate_up(bfloat16_t, bfloat16, 64, 1)
instantiate_warp_moe_gate_up(half, float16, 128, 1)
instantiate_warp_moe_gate_up(bfloat16_t, bfloat16, 128, 1)

// swiglu (GPT-OSS)
instantiate_warp_moe_gate_up(half, float16, 64, 2)
instantiate_warp_moe_gate_up(bfloat16_t, bfloat16, 64, 2)
instantiate_warp_moe_gate_up(half, float16, 128, 2)
instantiate_warp_moe_gate_up(bfloat16_t, bfloat16, 128, 2)
