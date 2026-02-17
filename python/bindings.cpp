#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "sintellix/codec/cic_data.hpp"
#include "sintellix/codec/vqgan.hpp"
#include "sintellix/codec/encoder.hpp"
#include "sintellix/codec/decoder.hpp"
#include "sintellix/core/config.hpp"
#include "sintellix/core/neuron_model.cuh"

// SXLM Phase 3 components
#include "sxlm/core/toml_config.hpp"
#include "sxlm/attention/hot_nsa.cuh"
#include "sxlm/memory/engram.cuh"
#include "sxlm/memory/sct.cuh"
#include "sxlm/training/el_trace.cuh"
#include "sxlm/training/mhc.cuh"
#include "sxlm/planning/ring_buffer.cuh"
#include "sxlm/multimodal/fusion.cuh"
#include "sxlm/tools/tool_port.cuh"

namespace py = pybind11;

PYBIND11_MODULE(_sintellix_native, m) {
    m.doc() = "Sintellix native C++/CUDA bindings";

    // CICData bindings
    py::class_<sintellix::CICData, std::shared_ptr<sintellix::CICData>>(m, "CICData")
        .def(py::init<>())
        .def_readwrite("src", &sintellix::CICData::src)
        .def_readwrite("emb", &sintellix::CICData::emb)
        .def_readwrite("metadata", &sintellix::CICData::metadata)
        .def_readwrite("nested", &sintellix::CICData::nested)
        .def("has_src", &sintellix::CICData::has_src)
        .def("has_emb", &sintellix::CICData::has_emb)
        .def("has_nested", &sintellix::CICData::has_nested)
        .def_static("create_with_src", &sintellix::CICData::create_with_src)
        .def_static("create_with_emb", &sintellix::CICData::create_with_emb)
        .def_static("create", &sintellix::CICData::create);

    // VQCodebook bindings
    py::class_<sintellix::VQCodebook, std::shared_ptr<sintellix::VQCodebook>>(m, "VQCodebook")
        .def(py::init<size_t, size_t>(), py::arg("codebook_size"), py::arg("embedding_dim"))
        .def("initialize", &sintellix::VQCodebook::initialize)
        .def("load_from_file", &sintellix::VQCodebook::load_from_file)
        .def("save_to_file", &sintellix::VQCodebook::save_to_file)
        .def("get_codebook_size", &sintellix::VQCodebook::get_codebook_size)
        .def("get_embedding_dim", &sintellix::VQCodebook::get_embedding_dim);

    // VQGANEncoder bindings
    py::class_<sintellix::VQGANEncoder>(m, "VQGANEncoder")
        .def(py::init<size_t, size_t, std::shared_ptr<sintellix::VQCodebook>>(),
             py::arg("input_dim"), py::arg("hidden_dim"), py::arg("codebook"))
        .def("initialize", &sintellix::VQGANEncoder::initialize)
        .def("get_input_dim", &sintellix::VQGANEncoder::get_input_dim);

    // VQGANDecoder bindings
    py::class_<sintellix::VQGANDecoder>(m, "VQGANDecoder")
        .def(py::init<size_t, size_t, std::shared_ptr<sintellix::VQCodebook>>(),
             py::arg("output_dim"), py::arg("hidden_dim"), py::arg("codebook"))
        .def("initialize", &sintellix::VQGANDecoder::initialize)
        .def("get_output_dim", &sintellix::VQGANDecoder::get_output_dim);

    // SemanticEncoder bindings
    py::class_<sintellix::SemanticEncoder>(m, "SemanticEncoder")
        .def(py::init<const std::string&, std::shared_ptr<sintellix::VQCodebook>>(),
             py::arg("model_path"), py::arg("codebook"))
        .def("initialize", &sintellix::SemanticEncoder::initialize)
        .def("encode", &sintellix::SemanticEncoder::encode)
        .def("encode_from_emb", &sintellix::SemanticEncoder::encode_from_emb)
        .def("encode_text", &sintellix::SemanticEncoder::encode_text)
        .def("encode_text_batch", &sintellix::SemanticEncoder::encode_text_batch);

    // SemanticDecoder bindings
    py::class_<sintellix::SemanticDecoder>(m, "SemanticDecoder")
        .def(py::init<std::shared_ptr<sintellix::VQCodebook>, size_t>(),
             py::arg("codebook"), py::arg("output_dim"))
        .def("initialize", &sintellix::SemanticDecoder::initialize)
        .def("decode", &sintellix::SemanticDecoder::decode)
        .def("decode_to_emb", &sintellix::SemanticDecoder::decode_to_emb)
        .def("decode_to_text", &sintellix::SemanticDecoder::decode_to_text)
        .def("get_output_dim", &sintellix::SemanticDecoder::get_output_dim);

    // ConfigManager bindings (disabled - requires Protobuf)
    // py::class_<sintellix::ConfigManager>(m, "ConfigManager")
    //     .def(py::init<>());

    // KFEManager bindings
    py::class_<sintellix::KFEManager>(m, "KFEManager")
        .def(py::init<size_t>(), py::arg("max_slots") = 10000)
        .def("has_kfe", &sintellix::KFEManager::has_kfe)
        .def("get_slot_count", &sintellix::KFEManager::get_slot_count);

    // NeuronModel bindings (disabled - requires Protobuf NeuronConfig)
    // py::class_<sintellix::NeuronModel>(m, "NeuronModel")
    //     .def(py::init<const sintellix::NeuronConfig&>());

    // ========== SXLM Phase 3 Components ==========

    // QuilaConfig (TOML-based)
    py::class_<sxlm::QuilaConfig>(m, "QuilaConfig")
        .def(py::init<>())
        .def_readwrite("dim", &sxlm::QuilaConfig::dim)
        .def_readwrite("num_heads", &sxlm::QuilaConfig::num_heads)
        .def_readwrite("num_layers", &sxlm::QuilaConfig::num_layers)
        .def_readwrite("learning_rate", &sxlm::QuilaConfig::learning_rate)
        .def_static("load", &sxlm::QuilaConfig::load)
        .def("save", &sxlm::QuilaConfig::save);

    // HOTConfig
    py::class_<sxlm::HOTConfig>(m, "HOTConfig")
        .def(py::init<>())
        .def_readwrite("dim", &sxlm::HOTConfig::dim)
        .def_readwrite("num_heads", &sxlm::HOTConfig::num_heads)
        .def_readwrite("hot_threshold", &sxlm::HOTConfig::hot_threshold);

    // EngramConfig
    py::class_<sxlm::EngramConfig>(m, "EngramConfig")
        .def(py::init<>())
        .def_readwrite("embedding_dim", &sxlm::EngramConfig::embedding_dim)
        .def_readwrite("num_hash_tables", &sxlm::EngramConfig::num_hash_tables);

    // MCPTool
    py::class_<sxlm::MCPTool>(m, "MCPTool")
        .def(py::init<>())
        .def_readwrite("name", &sxlm::MCPTool::name)
        .def_readwrite("description", &sxlm::MCPTool::description);
}
