#include "pipeline.cuh"
#include "../utils/memory.cuh"

namespace quila {

__host__ void init_pipeline(PipelineState* pipeline, int hidden_dim) {
    pipeline->current_phase = PHASE_0_ENCODING;
    pipeline->current_stage = STAGE_REPLAY;
    pipeline->replan_triggered = false;
    pipeline->hidden_dim = hidden_dim;
    pipeline->latent_z = (float*)allocate_unified(hidden_dim * sizeof(float));
}

__host__ void free_pipeline(PipelineState* pipeline) {
    if (pipeline->latent_z) {
        deallocate_unified(pipeline->latent_z);
    }
}

} // namespace quila
