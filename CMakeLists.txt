cmake_minimum_required(VERSION 3.20)
project(Quila LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

find_package(CUDAToolkit 12.0 REQUIRED)
find_package(Protobuf 3.21 REQUIRED)
find_package(OpenSSL REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${Protobuf_INCLUDE_DIRS}
    ${OPENSSL_INCLUDE_DIR}
)

# Protobuf generation
file(GLOB PROTO_FILES src/cpp/proto/*.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

# CUDA sources
file(GLOB_RECURSE CUDA_SOURCES src/cuda/**/*.cu)

# C++ sources
file(GLOB_RECURSE CPP_SOURCES src/cpp/**/*.cpp)

add_library(quila_core SHARED
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
    ${PROTO_SRCS}
)

target_link_libraries(quila_core
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    ${Protobuf_LIBRARIES}
    OpenSSL::SSL
    OpenSSL::Crypto
)

set_target_properties(quila_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
