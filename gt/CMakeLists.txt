cmake_minimum_required(VERSION 3.14.5)
project(lumi-benchmark LANGUAGES CXX)

find_package(MPI REQUIRED COMPONENTS CXX)

# User options
set(GTBENCH_BACKEND "x86" CACHE STRING "Benchmark backend")
set_property(CACHE GTBENCH_BACKEND PROPERTY STRINGS "x86" "mc" "cuda")

set(GTBENCH_FLOAT "double" CACHE STRING "Floating-point type")
set_property(CACHE GTBENCH_FLOAT PROPERTY STRINGS "float" "double")

set(GTBENCH_COMMUNICATION_BACKEND "ghex_comm" CACHE STRING "Communication backend")
set_property(CACHE GTBENCH_COMMUNICATION_BACKEND PROPERTY STRINGS "ghex_comm" "single_node" "simple_mpi")

if(GTBENCH_BACKEND STREQUAL "cuda")
  enable_language(CUDA)
endif()

# Helper functions
function(compile_as_cuda)
  get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  if("CUDA" IN_LIST languages)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CUDA)
  endif()
endfunction()

add_library(common INTERFACE)

# Global setup
if(GTBENCH_COMMUNICATION_BACKEND STREQUAL "ghex_comm")
  find_package(GHEX REQUIRED)
  target_link_libraries(common INTERFACE GHEX::ghexlib)
  target_compile_definitions(common INTERFACE GTBENCH_USE_GHEX)
endif()

target_compile_definitions(common INTERFACE
  GTBENCH_BACKEND=${GTBENCH_BACKEND}
  GTBENCH_FLOAT=${GTBENCH_FLOAT}
  GTBENCH_COMMUNICATION_BACKEND=${GTBENCH_COMMUNICATION_BACKEND}
  )
target_compile_options(common INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_60,code=sm_60>)

# Subdirectories
add_subdirectory(communication)
add_subdirectory(numerics)

# Current directory
compile_as_cuda(convergence_tests.cpp benchmark.cpp)

add_executable(convergence_tests convergence_tests.cpp)
target_link_libraries(convergence_tests advection diffusion communication)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark advection diffusion communication)
