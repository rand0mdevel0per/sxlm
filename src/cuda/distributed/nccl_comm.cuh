#pragma once
#include <cuda_runtime.h>

namespace quila {

// NCCL communication manager
struct NCCLComm {
    void** nccl_comms;  // NCCL communicators (opaque)
    int num_gpus;
    int rank;
};

// Initialize NCCL communication
__host__ void init_nccl_comm(NCCLComm* comm, int num_gpus, int rank);

// Free NCCL communication
__host__ void free_nccl_comm(NCCLComm* comm);

// P2P send to another GPU
__host__ void nccl_send(NCCLComm* comm, const void* data, size_t size, int dest_gpu);

// P2P receive from another GPU
__host__ void nccl_recv(NCCLComm* comm, void* data, size_t size, int src_gpu);

// AllReduce across all GPUs
__host__ void nccl_allreduce(NCCLComm* comm, const void* sendbuf, void* recvbuf, size_t count);

} // namespace quila
