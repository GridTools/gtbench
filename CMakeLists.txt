cmake_minimum_required(VERSION 3.14.5)
project(GTBench LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
endif()

# User options
set(GTBENCH_BACKEND "cpu_ifirst" CACHE STRING "Benchmark backend")
set_property(CACHE GTBENCH_BACKEND PROPERTY STRINGS "cpu_kfirst" "cpu_ifirst" "gpu")

set(GTBENCH_FLOAT "float" CACHE STRING "Floating-point type")
set_property(CACHE GTBENCH_FLOAT PROPERTY STRINGS "float" "double")

set(GTBENCH_RUNTIME "single_node" CACHE STRING "Runtime")
set_property(CACHE GTBENCH_RUNTIME PROPERTY STRINGS "single_node" "simple_mpi" "gcl" "ghex_comm")

find_package(GridTools 2.0 REQUIRED)

if(GTBENCH_BACKEND STREQUAL "cuda" AND (NOT DEFINED GRIDTOOLS_CUDA_COMPILATION_TYPE
                                        OR GRIDTOOLS_CUDA_COMPILATION_TYPE STREQUAL "NVCC-CUDA"))
  enable_language(CUDA)

  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_EXTENSIONS OFF)
endif()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

# Helper functions
function(compile_as_cuda)
  get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  if("CUDA" IN_LIST languages)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CUDA)
  endif()
endfunction()

add_library(common INTERFACE)
target_compile_options(common INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--default-stream per-thread>")

target_compile_definitions(common INTERFACE
  GTBENCH_BACKEND=${GTBENCH_BACKEND}
  GTBENCH_BACKEND_${GTBENCH_BACKEND}
  GTBENCH_FLOAT=${GTBENCH_FLOAT}
  )
target_link_libraries(common INTERFACE GridTools::gridtools GridTools::stencil_cpu_ifirst)

add_library(common_runtime INTERFACE)
target_compile_definitions(common_runtime INTERFACE
  GTBENCH_RUNTIME=${GTBENCH_RUNTIME}
  )
  target_link_libraries(common_runtime INTERFACE common)

# Subdirectories
add_subdirectory(common)
add_subdirectory(io)
add_subdirectory(runtime)
add_subdirectory(numerics)

# Current directory
compile_as_cuda(convergence_tests.cpp benchmark.cpp)

add_executable(convergence_tests convergence_tests.cpp)
target_link_libraries(convergence_tests advection diffusion options runtime device io)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark advection diffusion options runtime device io)
