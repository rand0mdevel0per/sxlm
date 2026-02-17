#include "sxlm/training/mhc.cuh"
#include <cmath>

namespace sxlm {

__global__ void project_manifold_kernel(
    double* gradients,
    const double* projection_matrix,
    int batch,
    int seq_len,
    int dim,
    float manifold_radius
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq_len;

    if (idx < total) {
        double* grad = gradients + idx * dim;

        // Compute norm
        double norm = 0.0;
        for (int i = 0; i < dim; i++) {
            norm += grad[i] * grad[i];
        }
        norm = sqrt(norm);

        // Project onto manifold sphere
        if (norm > manifold_radius) {
            double scale = manifold_radius / norm;
            for (int i = 0; i < dim; i++) {
                grad[i] *= scale;
            }
        }
    }
}

__global__ void hyperconnection_kernel(
    const double* input,
    const double* residual,
    double* output,
    int batch,
    int seq_len,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq_len * dim;

    if (idx < total) {
        output[idx] = input[idx] + residual[idx];
    }
}

ManifoldHyperconnection::ManifoldHyperconnection(const MHCConfig& config)
    : config_(config), projection_matrix_(nullptr) {
    cudaMalloc(&projection_matrix_, config.dim * config.dim * sizeof(double));
}

ManifoldHyperconnection::~ManifoldHyperconnection() {
    if (projection_matrix_) {
        cudaFree(projection_matrix_);
    }
}

void ManifoldHyperconnection::project_gradients(
    double* gradients,
    int batch,
    int seq_len
) {
    int total = batch * seq_len;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    project_manifold_kernel<<<blocks, threads>>>(
        gradients, projection_matrix_,
        batch, seq_len, config_.dim, config_.manifold_radius
    );
}

void ManifoldHyperconnection::forward(
    const double* input,
    const double* residual,
    double* output,
    int batch,
    int seq_len
) {
    int total = batch * seq_len * config_.dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    hyperconnection_kernel<<<blocks, threads>>>(
        input, residual, output,
        batch, seq_len, config_.dim
    );
}

} // namespace sxlm
