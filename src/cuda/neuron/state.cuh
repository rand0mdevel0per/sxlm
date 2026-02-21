#pragma once
#include <cuda_runtime.h>
#include <cuda_fp8.h>

namespace quila {

constexpr int ACTIVATION_HISTORY_SIZE = 16;
constexpr int CONTRIBUTION_HISTORY_SIZE = 32;
constexpr int TEMPORAL_FRAMES = 8;

// S1: Activation state
struct S1_Activation {
    float current_activation;
    float activation_history[ACTIVATION_HISTORY_SIZE];
    int history_index;
    int refractory_counter;
};

// S2: Computational state
struct S2_Computational {
    float* hidden_state;      // R^D
    float* ssm_state;         // R^D
    float* drc_accumulator;   // R^D
    float* residual_stream;   // R^D
};

// S3: Temporal state
struct S3_Temporal {
    float* frame_memory[TEMPORAL_FRAMES];  // 8 frames, each R^D
    float* momentum_vector;                 // R^D
};

// S4: Identity state
enum AttentionType { NSA, SSM, LINEAR_ATTN, DRC };
struct S4_Identity {
    AttentionType attention_type;
    float* receptive_field_mask;      // R^D
    float* specialization_vector;     // R^D
    int graph_position;
};

// S5: Confidence state
struct S5_Confidence {
    float output_confidence;
    float input_reliability;
    float* uncertainty_vector;  // R^D
    float calibration;
};

// S6: Connectivity state
struct S6_Connectivity {
    int* active_connections;
    int connection_count;
    int* routing_table;
    float* neighbour_summary;  // R^D
    __nv_fp8_e4m3* outgoing_message_fp8;  // fp8 E4M3 encoded (Req 5.3.2)
    __nv_fp8_e4m3* incoming_message_fp8;  // fp8 E4M3 encoded (Req 5.3.2)
};

// S7: Plasticity state
struct S7_Plasticity {
    float* adam_m;              // R^D
    float* adam_v;              // R^D
    float* persistent_el_trace; // R^D
    float learnable_lambda;
};

// S8: Utility state
struct S8_Utility {
    float utility_ema;
    float contribution_history[CONTRIBUTION_HISTORY_SIZE];
    int history_index;
    float eviction_priority;
};

// Complete neuron state
struct NeuronState {
    S1_Activation s1;
    S2_Computational s2;
    S3_Temporal s3;
    S4_Identity s4;
    S5_Confidence s5;
    S6_Connectivity s6;
    S7_Plasticity s7;
    S8_Utility s8;
    int hidden_dim;
};

} // namespace quila
