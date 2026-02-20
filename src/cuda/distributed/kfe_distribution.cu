#include "kfe_distribution.cuh"
#include "../utils/memory.cuh"

namespace quila {

__host__ void init_kfe_distribution(KFEDistribution* dist, int num_kfes, int num_gpus) {
    dist->num_kfes = num_kfes;
    dist->num_gpus = num_gpus;

    dist->kfe_to_gpu = (int*)allocate_unified(num_kfes * sizeof(int));
    dist->gpu_kfe_counts = (int*)allocate_unified(num_gpus * sizeof(int));

    cudaMemset(dist->gpu_kfe_counts, 0, num_gpus * sizeof(int));
}

__host__ void free_kfe_distribution(KFEDistribution* dist) {
    if (dist->kfe_to_gpu) deallocate_unified(dist->kfe_to_gpu);
    if (dist->gpu_kfe_counts) deallocate_unified(dist->gpu_kfe_counts);
}

__host__ void assign_kfes_to_gpus(KFEDistribution* dist) {
    // Simplified: round-robin assignment
    for (int i = 0; i < dist->num_kfes; i++) {
        int gpu_id = i % dist->num_gpus;
        dist->kfe_to_gpu[i] = gpu_id;
        dist->gpu_kfe_counts[gpu_id]++;
    }
}

__host__ int get_kfe_gpu(const KFEDistribution* dist, int kfe_id) {
    return dist->kfe_to_gpu[kfe_id];
}

} // namespace quila
