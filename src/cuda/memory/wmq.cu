#include "wmq.cuh"
#include "../utils/memory.cuh"
#include "../utils/error.cuh"

namespace quila {

__host__ void init_wmq(WMQ* wmq, int num_stages, int hidden_dim, int stage_capacity) {
    wmq->num_stages = num_stages;
    wmq->hidden_dim = hidden_dim;
    wmq->stages = (WMQStage*)allocate_unified(num_stages * sizeof(WMQStage));

    for (int i = 0; i < num_stages; i++) {
        wmq->stages[i].data = (float*)allocate_unified(stage_capacity * hidden_dim * sizeof(float));
        wmq->stages[i].capacity = stage_capacity;
        wmq->stages[i].head = 0;
        wmq->stages[i].tail = 0;
        wmq->stages[i].size = 0;
        wmq->stages[i].is_active = false;
    }
}

__host__ void free_wmq(WMQ* wmq) {
    for (int i = 0; i < wmq->num_stages; i++) {
        deallocate_unified(wmq->stages[i].data);
    }
    deallocate_unified(wmq->stages);
}

__device__ void wmq_push(WMQStage* stage, const float* data, int hidden_dim) {
    if (stage->size < stage->capacity) {
        int offset = stage->head * hidden_dim;
        for (int i = 0; i < hidden_dim; i++) {
            stage->data[offset + i] = data[i];
        }
        stage->head = (stage->head + 1) % stage->capacity;
        stage->size++;
    }
}

__device__ void wmq_pop(WMQStage* stage, float* data, int hidden_dim) {
    if (stage->size > 0) {
        int offset = stage->tail * hidden_dim;
        for (int i = 0; i < hidden_dim; i++) {
            data[i] = stage->data[offset + i];
        }
        stage->tail = (stage->tail + 1) % stage->capacity;
        stage->size--;
    }
}

} // namespace quila
