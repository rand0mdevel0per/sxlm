#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "cuda/neuron/forward.cuh"
#include "cuda/memory/wmq.cuh"
#include "cuda/inference/pipeline.cuh"

namespace py = pybind11;

class QuilaModel {
private:
    NeuronState* neuron_states;
    quila::WMQ wmq;
    quila::PipelineState pipeline;
    int num_neurons;
    int hidden_dim;

public:
    QuilaModel(int num_neurons = 32768, int hidden_dim = 256)
        : num_neurons(num_neurons), hidden_dim(hidden_dim) {
        cudaMallocManaged(&neuron_states, num_neurons * sizeof(NeuronState));
        for (int i = 0; i < num_neurons; i++) {
            neuron_states[i].hidden_dim = hidden_dim;
            cudaMalloc(&neuron_states[i].s1_semantic, hidden_dim * sizeof(float));
            cudaMalloc(&neuron_states[i].s2_episodic, hidden_dim * sizeof(float));
            cudaMalloc(&neuron_states[i].s3_working, hidden_dim * sizeof(float));
            cudaMalloc(&neuron_states[i].s4_plan, hidden_dim * sizeof(float));
            cudaMalloc(&neuron_states[i].s5_tool, hidden_dim * sizeof(float));
            cudaMalloc(&neuron_states[i].s6_output, hidden_dim * sizeof(float));
            cudaMalloc(&neuron_states[i].s7_conflict, hidden_dim * sizeof(float));
            cudaMalloc(&neuron_states[i].s8_meta, hidden_dim * sizeof(float));
        }
        quila::init_wmq(&wmq, 6, hidden_dim, 128);
        quila::init_pipeline(&pipeline, hidden_dim);
    }

    ~QuilaModel() {
        for (int i = 0; i < num_neurons; i++) {
            cudaFree(neuron_states[i].s1_semantic);
            cudaFree(neuron_states[i].s2_episodic);
            cudaFree(neuron_states[i].s3_working);
            cudaFree(neuron_states[i].s4_plan);
            cudaFree(neuron_states[i].s5_tool);
            cudaFree(neuron_states[i].s6_output);
            cudaFree(neuron_states[i].s7_conflict);
            cudaFree(neuron_states[i].s8_meta);
        }
        cudaFree(neuron_states);
        quila::free_wmq(&wmq);
        quila::free_pipeline(&pipeline);
    }

    std::string inference(const std::string& prompt) {
        return "Quila response to: " + prompt;
    }

    py::dict get_status() {
        py::dict status;
        status["neurons"] = num_neurons;
        status["hidden_dim"] = hidden_dim;
        status["phase"] = static_cast<int>(pipeline.current_phase);
        return status;
    }
};

PYBIND11_MODULE(quila_core, m) {
    py::class_<QuilaModel>(m, "QuilaModel")
        .def(py::init<int, int>(), py::arg("num_neurons") = 32768, py::arg("hidden_dim") = 256)
        .def("inference", &QuilaModel::inference)
        .def("get_status", &QuilaModel::get_status);
}
