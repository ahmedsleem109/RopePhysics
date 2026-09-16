cmake_minimum_required(VERSION 3.20)
project(gpu_cosserat_rod LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_library(crs_core
  src/core/collision.cpp
  src/core/coloring.cpp
  src/core/rod.cpp
  src/core/solver.cpp
  src/core/statics.cpp
  src/apps/cable_hanging.cpp
  src/apps/harness_routing.cpp
)
target_include_directories(crs_core PUBLIC src)

add_executable(rodsim
  src/main.cpp
  src/validation/cases.cpp
  src/validation/contact_cases.cpp
  src/validation/scaling_cases.cpp
  src/validation/scenes.cpp
  src/validation/application_cases.cpp
  src/gpu/gpu_cases.cpp
)
target_link_libraries(rodsim PRIVATE crs_core)

if(MSVC)
  target_compile_options(crs_core PRIVATE /W4 /fp:precise)
  target_compile_options(rodsim PRIVATE /W4 /fp:precise)
else()
  target_compile_options(crs_core PRIVATE -Wall -Wextra)
  target_compile_options(rodsim PRIVATE -Wall -Wextra)
endif()

# ---------------------------------------------------------------- CUDA
# Optional: the CPU reference and its whole validation suite build and run
# without it. When a CUDA compiler is present the GPU solver and its parity,
# determinism and throughput cases are compiled into the same rodsim binary.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3060 (Ampere)
  endif()

  # Only the .cu goes through nvcc; all host orchestration is ordinary C++.
  add_library(crs_gpu src/gpu/gpu_solver.cu src/gpu/gpu_batch.cpp)
  target_link_libraries(crs_gpu PUBLIC crs_core)
  target_include_directories(crs_gpu PUBLIC src)

  target_link_libraries(rodsim PRIVATE crs_gpu)
  target_compile_definitions(rodsim PRIVATE CRS_WITH_CUDA)
  message(STATUS "CUDA enabled (${CMAKE_CUDA_COMPILER}), arch ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "CUDA compiler not found - building the CPU reference only")
endif()

# Exploratory probes: tools for deciding what a validation case should test,
# not validation cases themselves.
add_executable(rodexp tools/experiments/stability_probe.cpp)
target_link_libraries(rodexp PRIVATE crs_core)

enable_testing()
add_test(NAME validation COMMAND rodsim all --out ${CMAKE_SOURCE_DIR}/docs/data)
