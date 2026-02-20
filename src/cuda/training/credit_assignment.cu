#include "credit_assignment.cuh"
#include "../utils/memory.cuh"
#include "../utils/error.cuh"

namespace quila {

__host__ void init_credit_assignment(CreditAssignment* ca, int num_neurons) {
    ca->num_neurons = num_neurons;
    ca->neuron_credits = (float*)allocate_unified(num_neurons * sizeof(float));
    cudaMemset(ca->neuron_credits, 0, num_neurons * sizeof(float));
}

__host__ void free_credit_assignment(CreditAssignment* ca) {
    if (ca->neuron_credits) deallocate_unified(ca->neuron_credits);
}

__global__ void assign_credit_kernel(
    float* neuron_credits,
    const float* session_trace,
    const float* persistent_trace,
    float reward,
    int num_neurons
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_neurons) return;

    neuron_credits[tid] = reward * (session_trace[tid] + 0.1f * persistent_trace[tid]);
}

__host__ void assign_credit(CreditAssignment* ca, const ElTrace* trace, float reward) {
    int threads = 256;
    int blocks = (ca->num_neurons + threads - 1) / threads;

    assign_credit_kernel<<<blocks, threads>>>(
        ca->neuron_credits, trace->session_trace, trace->persistent_trace, reward, ca->num_neurons
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace quila
