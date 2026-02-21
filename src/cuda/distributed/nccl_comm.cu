#include "nccl_comm.cuh"
#include "../utils/error.cuh"

namespace quila {

__host__ void init_nccl_comm(NCCLComm* comm, int num_gpus, int rank) {
    comm->num_gpus = num_gpus;
    comm->rank = rank;
    comm->nccl_comms = nullptr;  // Simplified: NCCL not initialized
}

__host__ void free_nccl_comm(NCCLComm* comm) {
    // Simplified: no cleanup needed
    (void)comm;
}

__host__ void nccl_send(NCCLComm* comm, const void* data, size_t size, int dest_gpu) {
    // IMPORTANT: Do NOT use explicit cudaMemcpy between GPUs (Req 22.1.3)
    // Use CUDA Unified Memory + NVLink P2P instead
    (void)comm; (void)data; (void)size; (void)dest_gpu;
}

__host__ void nccl_recv(NCCLComm* comm, void* data, size_t size, int src_gpu) {
    // IMPORTANT: Do NOT use explicit cudaMemcpy between GPUs (Req 22.1.3)
    // Use CUDA Unified Memory + NVLink P2P instead
    (void)comm; (void)data; (void)size; (void)src_gpu;
}

__host__ void nccl_allreduce(NCCLComm* comm, const void* sendbuf, void* recvbuf, size_t count) {
    // IMPORTANT: Do NOT use explicit cudaMemcpy between GPUs (Req 22.1.3)
    // Use CUDA Unified Memory + NVLink P2P instead
    // TODO: When implementing, use fp32 accumulation for all reductions (Req 5.3.1)
    cudaMemcpy(recvbuf, sendbuf, count, cudaMemcpyDeviceToDevice);
    (void)comm;
}

} // namespace quila
