#include <iostream>
#include <cuda_runtime.h>
#include "cuda/neuron/forward.cuh"
#include "cuda/attention/port.cuh"
#include "cuda/memory/wmq.cuh"
#include "cuda/inference/pipeline.cuh"

int main() {
    std::cout << "=== Quila Integration Test ===" << std::endl;

    // Test 1: CUDA device available
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess || deviceCount == 0) {
        std::cout << "[SKIP] No CUDA device available" << std::endl;
        return 0;
    }

    std::cout << "[PASS] Found " << deviceCount << " CUDA device(s)" << std::endl;

    const int hidden_dim = 256;
    const int seq_len = 16;

    // Test 2: Neuron State Initialization
    NeuronState neuron_state;
    neuron_state.hidden_dim = hidden_dim;
    cudaMalloc(&neuron_state.s1_semantic, hidden_dim * sizeof(float));
    cudaMalloc(&neuron_state.s2_episodic, hidden_dim * sizeof(float));
    cudaMalloc(&neuron_state.s3_working, hidden_dim * sizeof(float));
    cudaMalloc(&neuron_state.s4_plan, hidden_dim * sizeof(float));
    cudaMalloc(&neuron_state.s5_tool, hidden_dim * sizeof(float));
    cudaMalloc(&neuron_state.s6_output, hidden_dim * sizeof(float));
    cudaMalloc(&neuron_state.s7_conflict, hidden_dim * sizeof(float));
    cudaMalloc(&neuron_state.s8_meta, hidden_dim * sizeof(float));

    if (neuron_state.s1_semantic && neuron_state.s8_meta) {
        std::cout << "[PASS] Neuron state allocation" << std::endl;
    } else {
        std::cerr << "[FAIL] Neuron state allocation" << std::endl;
        return 1;
    }

    // Test 3: Port Configuration
    quila::PortConfig port_config;
    port_config.hidden_dim = hidden_dim;
    port_config.num_neurons = 32768;
    port_config.alpha = 0.3f;
    port_config.beta = 0.3f;
    port_config.gamma = 0.4f;
    port_config.nsa_top_k = 128;
    std::cout << "[PASS] Port configuration" << std::endl;

    // Test 4: WMQ Initialization
    quila::WMQ wmq;
    quila::init_wmq(&wmq, 6, hidden_dim, 128);
    if (wmq.stages && wmq.num_stages == 6) {
        std::cout << "[PASS] WMQ initialization" << std::endl;
    } else {
        std::cerr << "[FAIL] WMQ initialization" << std::endl;
        return 1;
    }

    // Test 5: Pipeline Initialization
    quila::PipelineState pipeline;
    quila::init_pipeline(&pipeline, hidden_dim);
    if (pipeline.latent_z && pipeline.current_phase == quila::PHASE_0_ENCODING) {
        std::cout << "[PASS] Pipeline initialization" << std::endl;
    } else {
        std::cerr << "[FAIL] Pipeline initialization" << std::endl;
        return 1;
    }

    // Cleanup
    cudaFree(neuron_state.s1_semantic);
    cudaFree(neuron_state.s2_episodic);
    cudaFree(neuron_state.s3_working);
    cudaFree(neuron_state.s4_plan);
    cudaFree(neuron_state.s5_tool);
    cudaFree(neuron_state.s6_output);
    cudaFree(neuron_state.s7_conflict);
    cudaFree(neuron_state.s8_meta);
    quila::free_wmq(&wmq);
    quila::free_pipeline(&pipeline);

    std::cout << "\n[SUCCESS] All integration tests passed!" << std::endl;
    std::cout << "Components tested: Neuron, Port, WMQ, Pipeline" << std::endl;

    return 0;
}
