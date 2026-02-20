#pragma once
#include <cuda_runtime.h>

namespace quila {

// Inference phase enum
enum InferencePhase {
    PHASE_0_ENCODING = 0,
    PHASE_1_REPLAY = 1,
    PHASE_2_LASTINPUT = 2,
    PHASE_3_REREAD = 3,
    PHASE_4_PLAN = 4,
    PHASE_5_THINK = 5,
    PHASE_6_OUTPUT = 6
};

// Stage type for WMQ management
enum StageType {
    STAGE_REPLAY,
    STAGE_LASTINPUT,
    STAGE_REREAD,
    STAGE_PLAN,
    STAGE_THINK,
    STAGE_OUTPUT
};

// Pipeline state
struct PipelineState {
    InferencePhase current_phase;
    StageType current_stage;
    bool replan_triggered;
    float* latent_z;  // Plan latent vector
    int hidden_dim;
};

// Initialize pipeline
__host__ void init_pipeline(PipelineState* pipeline, int hidden_dim);

// Free pipeline
__host__ void free_pipeline(PipelineState* pipeline);

} // namespace quila
