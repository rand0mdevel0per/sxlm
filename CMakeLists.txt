cmake_minimum_required(VERSION 3.20)
project(Quila LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

find_package(CUDAToolkit 12.0 REQUIRED)
find_package(Protobuf 3.21)

include_directories(${CMAKE_SOURCE_DIR}/src)

if(Protobuf_FOUND)
    include_directories(${PROTOBUF_INCLUDE_DIRS})
    file(GLOB PROTO_FILES src/cpp/proto/*.proto)
    protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
endif()

file(GLOB_RECURSE CUDA_SOURCES src/cuda/**/*.cu)
file(GLOB_RECURSE CPP_SOURCES src/cpp/**/*.cpp)

add_library(quila_core SHARED ${CUDA_SOURCES} ${CPP_SOURCES} ${PROTO_SRCS})

target_link_libraries(quila_core CUDA::cudart CUDA::cublas)
if(Protobuf_FOUND)
    target_link_libraries(quila_core ${Protobuf_LIBRARIES})
endif()

set_target_properties(quila_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
