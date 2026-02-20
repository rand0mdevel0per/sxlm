#pragma once
#include <cuda_runtime.h>
#include "state.cuh"
#include "streams/nsa.cu"
#include "streams/ssm.cu"
#include "streams/linear_attn.cu"
#include "streams/drc.cu"
#include "mhc.cu"
#include "consistency_model.cu"
#include "../utils/error.cuh"

namespace quila {

// Minimal 12-step neuron forward pass
__global__ void neuron_forward_kernel(
    NeuronState* states,
    const float* input_messages,
    float* output_messages,
    int num_neurons,
    int hidden_dim
) {
    int neuron_id = blockIdx.x;
    if (neuron_id >= num_neurons) return;

    NeuronState& state = states[neuron_id];
    int tid = threadIdx.x;

    // Shared memory for intermediate results
    extern __shared__ float shared_mem[];
    float* h = shared_mem;
    float* out_a = h + hidden_dim;
    float* out_b = out_a + hidden_dim;
    float* out_c = out_b + hidden_dim;
    float* out_d = out_c + hidden_dim;
    float* h_mid = out_d + hidden_dim;
    float* h_clean = h_mid + hidden_dim;

    // STEP 1: Input aggregation (simplified)
    if (tid < hidden_dim) {
        h[tid] = input_messages[neuron_id * hidden_dim + tid];
    }
    __syncthreads();

    // STEP 2: Refractory check
    if (state.s1.refractory_counter > 0) {
        state.s1.refractory_counter--;
        return;  // Skip computation
    }

    // STEP 3: Receptive field filter (simplified - skip for minimal version)

    // STEP 4: Four parallel computation streams
    stream_a_nsa(h, nullptr, out_a, hidden_dim, hidden_dim / 2);
    __syncthreads();

    stream_b_ssm(h, nullptr, state.s2.ssm_state, out_b, hidden_dim);
    __syncthreads();

    stream_c_linear_attn(h, state.s2.ssm_state, out_c, hidden_dim);
    __syncthreads();

    stream_d_drc(h, state.s5.output_confidence, out_d, hidden_dim);
    __syncthreads();

    // STEP 5: mHC mixing
    mhc_mix(out_a, out_b, out_c, out_d, state.s4.specialization_vector, h_mid, hidden_dim);
    __syncthreads();

    // STEP 6: KFE recall (simplified - skip for minimal version)

    // STEP 7: Consistency Model denoising
    consistency_model_denoise(h_mid, state.s5.output_confidence, h_clean, hidden_dim);
    __syncthreads();

    // STEP 8: FXAA temporal smoothing (simplified)
    if (tid < hidden_dim) {
        float alpha = 0.3f;
        h_clean[tid] = (1.0f - alpha) * h_clean[tid] + alpha * state.s3.frame_memory[0][tid];
    }
    __syncthreads();

    // STEP 9: State updates
    if (tid == 0) {
        state.s5.output_confidence = 1.0f / (1.0f + expf(-h_clean[0]));
        state.s1.refractory_counter = 4 - (int)(state.s5.output_confidence * 3);
    }

    // STEP 10: KFE generalization (skip for minimal version)

    // STEP 11: Thermal noise
    if (tid < hidden_dim) {
        float sigma = 0.01f * (1.0f - state.s5.output_confidence);
        // Simplified noise (would use curand in full version)
        h_clean[tid] += sigma * 0.1f;
    }
    __syncthreads();

    // STEP 12: Output routing
    if (tid < hidden_dim) {
        output_messages[neuron_id * hidden_dim + tid] = h_clean[tid];
    }
}

} // namespace quila
