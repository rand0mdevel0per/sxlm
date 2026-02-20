#pragma once
#include "state.cuh"
#include "../utils/memory.cu"
#include "../utils/error.cuh"

namespace quila {

// Initialize neuron state with memory allocation
__host__ void init_neuron_state(NeuronState* state, int hidden_dim) {
    state->hidden_dim = hidden_dim;

    // S1: Activation
    state->s1.current_activation = 0.0f;
    state->s1.history_index = 0;
    state->s1.refractory_counter = 0;
    for (int i = 0; i < ACTIVATION_HISTORY_SIZE; i++) {
        state->s1.activation_history[i] = 0.0f;
    }

    // S2: Computational
    state->s2.hidden_state = (float*)allocate_unified(hidden_dim * sizeof(float));
    state->s2.ssm_state = (float*)allocate_unified(hidden_dim * sizeof(float));
    state->s2.drc_accumulator = (float*)allocate_unified(hidden_dim * sizeof(float));
    state->s2.residual_stream = (float*)allocate_unified(hidden_dim * sizeof(float));

    // S3: Temporal
    for (int i = 0; i < TEMPORAL_FRAMES; i++) {
        state->s3.frame_memory[i] = (float*)allocate_unified(hidden_dim * sizeof(float));
    }
    state->s3.momentum_vector = (float*)allocate_unified(hidden_dim * sizeof(float));

    // S4: Identity
    state->s4.attention_type = NSA;
    state->s4.receptive_field_mask = (float*)allocate_unified(hidden_dim * sizeof(float));
    state->s4.specialization_vector = (float*)allocate_unified(hidden_dim * sizeof(float));
    state->s4.graph_position = 0;

    // S5: Confidence
    state->s5.output_confidence = 0.5f;
    state->s5.input_reliability = 1.0f;
    state->s5.uncertainty_vector = (float*)allocate_unified(hidden_dim * sizeof(float));
    state->s5.calibration = 1.0f;

    // S6: Connectivity
    state->s6.connection_count = 0;
    state->s6.active_connections = nullptr;
    state->s6.routing_table = nullptr;
    state->s6.neighbour_summary = (float*)allocate_unified(hidden_dim * sizeof(float));

    // S7: Plasticity
    state->s7.adam_m = (float*)allocate_unified(hidden_dim * sizeof(float));
    state->s7.adam_v = (float*)allocate_unified(hidden_dim * sizeof(float));
    state->s7.persistent_el_trace = (float*)allocate_unified(hidden_dim * sizeof(float));
    state->s7.learnable_lambda = 0.9f;

    // S8: Utility
    state->s8.utility_ema = 0.0f;
    state->s8.history_index = 0;
    state->s8.eviction_priority = 0.0f;
    for (int i = 0; i < CONTRIBUTION_HISTORY_SIZE; i++) {
        state->s8.contribution_history[i] = 0.0f;
    }
}

// Free neuron state memory
__host__ void free_neuron_state(NeuronState* state) {
    deallocate_unified(state->s2.hidden_state);
    deallocate_unified(state->s2.ssm_state);
    deallocate_unified(state->s2.drc_accumulator);
    deallocate_unified(state->s2.residual_stream);

    for (int i = 0; i < TEMPORAL_FRAMES; i++) {
        deallocate_unified(state->s3.frame_memory[i]);
    }
    deallocate_unified(state->s3.momentum_vector);

    deallocate_unified(state->s4.receptive_field_mask);
    deallocate_unified(state->s4.specialization_vector);
    deallocate_unified(state->s5.uncertainty_vector);
    deallocate_unified(state->s6.neighbour_summary);
    deallocate_unified(state->s7.adam_m);
    deallocate_unified(state->s7.adam_v);
    deallocate_unified(state->s7.persistent_el_trace);
}

} // namespace quila
