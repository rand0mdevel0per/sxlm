#include "consistency_distillation.cuh"
#include "../utils/error.cuh"

namespace quila {

__host__ void init_consistency_trainer(
    ConsistencyModelTrainer* trainer,
    int hidden_dim,
    float learning_rate
) {
    trainer->hidden_dim = hidden_dim;
    trainer->learning_rate = learning_rate;
    CHECK_CUDA_ERROR(cudaMalloc(&trainer->teacher_params, hidden_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&trainer->student_params, hidden_dim * sizeof(float)));
}

__global__ void consistency_distillation_kernel(
    const float* teacher_params,
    float* student_params,
    const float* noisy_input,
    float timestep,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden_dim) return;

    // Consistency distillation: match teacher output at t and t-1
    float teacher_out = teacher_params[idx] * expf(-timestep);
    float student_out = student_params[idx];
    float loss_grad = 2.0f * (student_out - teacher_out);

    // Update student parameters
    student_params[idx] -= 0.001f * loss_grad;
}

__host__ float train_consistency_step(
    ConsistencyModelTrainer* trainer,
    const float* noisy_input,
    float timestep,
    int batch_size
) {
    int threads = 256;
    int blocks = (trainer->hidden_dim + threads - 1) / threads;

    consistency_distillation_kernel<<<blocks, threads>>>(
        trainer->teacher_params,
        trainer->student_params,
        noisy_input,
        timestep,
        trainer->hidden_dim
    );

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return 0.0f;  // Return dummy loss
}

} // namespace quila
