#pragma once

#include <cuda_runtime.h>

namespace sxlm {

// mHC: Manifold-constrained Hyperconnections
// Stabilizes gradients through manifold projection

struct MHCConfig {
    int dim;                    // Model dimension
    float manifold_radius;      // Manifold constraint radius (e.g., 1.0)
    float projection_strength;  // Projection strength (e.g., 0.1)
    bool enable_orthogonal;     // Enable orthogonal constraint
};

class ManifoldHyperconnection {
public:
    ManifoldHyperconnection(const MHCConfig& config);
    ~ManifoldHyperconnection();

    // Project gradients onto manifold
    void project_gradients(
        double* gradients,
        int batch,
        int seq_len
    );

    // Apply hyperconnection with manifold constraint
    void forward(
        const double* input,
        const double* residual,
        double* output,
        int batch,
        int seq_len
    );

private:
    MHCConfig config_;
    double* projection_matrix_;  // Projection matrix for manifold
};

} // namespace sxlm
