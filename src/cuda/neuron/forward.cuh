#pragma once
#include <cuda_runtime.h>

struct NeuronState {
    float* s1_semantic;
    float* s2_episodic;
    float* s3_working;
    float* s4_plan;
    float* s5_tool;
    float* s6_output;
    float* s7_conflict;
    float* s8_meta;
    int hidden_dim;
};

__host__ void neuron_forward(NeuronState* state, const float* input, float* output, int seq_len);
