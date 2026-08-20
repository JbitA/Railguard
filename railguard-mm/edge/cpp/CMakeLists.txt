cmake_minimum_required(VERSION 3.24)
project(railguard_edge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RAILGUARD_ENABLE_CUDA "Enable CUDA preprocessing" OFF)
option(RAILGUARD_ENABLE_TENSORRT "Enable TensorRT inference" OFF)
option(RAILGUARD_BUILD_BENCHMARKS "Build microbenchmarks" ON)
option(RAILGUARD_WARNINGS_AS_ERRORS "Treat native warnings as errors" OFF)
option(RAILGUARD_ENABLE_SANITIZERS "Enable AddressSanitizer + UndefinedBehaviorSanitizer for host validation" OFF)

add_library(railguard_core
    src/protocol.cpp
    src/dsp.cpp
    src/runtime.cpp
    src/reference_inference.cpp
    src/serial_source.cpp
    src/v4l2_camera.cpp
    src/synchronizer.cpp
    src/spatial_fusion.cpp
    src/clock_alignment.cpp
    src/resampler.cpp
    src/image_quality.cpp
    src/camera_preprocess.cpp
)
target_include_directories(railguard_core PUBLIC include)
target_compile_options(railguard_core PRIVATE -Wall -Wextra -Wpedantic)
if(RAILGUARD_WARNINGS_AS_ERRORS)
    target_compile_options(railguard_core PRIVATE -Werror)
endif()
if(RAILGUARD_ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(railguard_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(railguard_core PUBLIC -fsanitize=address,undefined)
    else()
        message(FATAL_ERROR "RAILGUARD_ENABLE_SANITIZERS currently supports GCC/Clang host builds")
    endif()
endif()

if(RAILGUARD_ENABLE_CUDA)
    enable_language(CUDA)
    add_library(railguard_cuda cuda/preprocess.cu)
    target_include_directories(railguard_cuda PUBLIC include)
    target_compile_definitions(railguard_core PUBLIC RAILGUARD_ENABLE_CUDA=1)
    target_link_libraries(railguard_core PUBLIC railguard_cuda)
endif()

if(RAILGUARD_ENABLE_TENSORRT)
    find_package(CUDAToolkit REQUIRED)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS /usr/include /usr/include/aarch64-linux-gnu /usr/local/TensorRT/include)
    find_library(TENSORRT_LIB nvinfer HINTS /usr/lib /usr/lib/aarch64-linux-gnu /usr/local/TensorRT/lib)
    find_library(TENSORRT_ONNX_LIB nvonnxparser HINTS /usr/lib /usr/lib/aarch64-linux-gnu /usr/local/TensorRT/lib)
    if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB)
        message(FATAL_ERROR "TensorRT requested but NvInfer.h/libnvinfer were not found")
    endif()
    target_sources(railguard_core PRIVATE src/tensorrt_engine.cpp)
    target_include_directories(railguard_core PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(railguard_core PRIVATE ${TENSORRT_LIB} ${TENSORRT_ONNX_LIB} CUDA::cudart)
    target_compile_definitions(railguard_core PUBLIC RAILGUARD_ENABLE_TENSORRT=1)
endif()

add_executable(railguard_edge src/main.cpp)
target_link_libraries(railguard_edge PRIVATE railguard_core)

if(RAILGUARD_BUILD_BENCHMARKS)
    add_executable(railguard_bench benchmarks/edge_bench.cpp)
    target_link_libraries(railguard_bench PRIVATE railguard_core)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(test_protocol tests/test_protocol.cpp)
  target_link_libraries(test_protocol PRIVATE railguard_core)
  add_test(NAME protocol COMMAND test_protocol)
  add_executable(test_dsp tests/test_dsp.cpp)
  target_link_libraries(test_dsp PRIVATE railguard_core)
  add_test(NAME dsp COMMAND test_dsp)
  add_executable(test_synchronizer tests/test_synchronizer.cpp)
  target_link_libraries(test_synchronizer PRIVATE railguard_core)
  add_test(NAME synchronizer COMMAND test_synchronizer)
  add_executable(test_spatial_fusion tests/test_spatial_fusion.cpp)
  target_link_libraries(test_spatial_fusion PRIVATE railguard_core)
  add_test(NAME spatial_fusion COMMAND test_spatial_fusion)
  add_executable(test_clock_alignment tests/test_clock_alignment.cpp)
  target_link_libraries(test_clock_alignment PRIVATE railguard_core)
  add_test(NAME clock_alignment COMMAND test_clock_alignment)
  add_executable(test_resampler tests/test_resampler.cpp)
  target_link_libraries(test_resampler PRIVATE railguard_core)
  add_test(NAME resampler COMMAND test_resampler)
  add_executable(test_image_quality tests/test_image_quality.cpp)
  target_link_libraries(test_image_quality PRIVATE railguard_core)
  add_test(NAME image_quality COMMAND test_image_quality)
  add_executable(test_camera_preprocess tests/test_camera_preprocess.cpp)
  target_link_libraries(test_camera_preprocess PRIVATE railguard_core)
  add_test(NAME camera_preprocess COMMAND test_camera_preprocess)
  add_executable(test_inference_contract tests/test_inference_contract.cpp)
  target_link_libraries(test_inference_contract PRIVATE railguard_core)
  add_test(NAME inference_contract COMMAND test_inference_contract)
endif()
