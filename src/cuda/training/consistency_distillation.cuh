#pragma once
#include <cuda_runtime.h>

namespace quila {

// Consistency Model training via Consistency Distillation (Req 6.3.1)
struct ConsistencyModelTrainer {
    float* teacher_params;
    float* student_params;
    int hidden_dim;
    float learning_rate;
};

// Initialize consistency model trainer
__host__ void init_consistency_trainer(
    ConsistencyModelTrainer* trainer,
    int hidden_dim,
    float learning_rate
);

// Train one step via consistency distillation
__host__ float train_consistency_step(
    ConsistencyModelTrainer* trainer,
    const float* noisy_input,
    float timestep,
    int batch_size
);

} // namespace quila
