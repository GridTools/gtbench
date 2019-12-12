cmake_minimum_required(VERSION 3.14.5)
project(lumi-benchmark LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
endif()

# User options
set(GTBENCH_BACKEND "x86" CACHE STRING "Benchmark backend")
set_property(CACHE GTBENCH_BACKEND PROPERTY STRINGS "x86" "mc" "cuda")

set(GTBENCH_FLOAT "float" CACHE STRING "Floating-point type")
set_property(CACHE GTBENCH_FLOAT PROPERTY STRINGS "float" "double")

set(GTBENCH_COMMUNICATION_BACKEND "ghex_comm" CACHE STRING "Communication backend")
set_property(CACHE GTBENCH_COMMUNICATION_BACKEND PROPERTY STRINGS "ghex_comm" "single_node" "simple_mpi")

set(GTBENCH_CUDA_ARCH "sm_60" CACHE STRING "Cuda architecture")

find_package(GridTools 1.1 REQUIRED)

set(LUMI_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${LUMI_MODULE_PATH}")

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

# Global setup
if(GTBENCH_COMMUNICATION_BACKEND STREQUAL "ghex_comm")
  find_package(GHEX REQUIRED)
  target_link_libraries(common INTERFACE GHEX::ghexlib)
  if (GHEX_USE_UCP)
      find_package(UCP REQUIRED)
      target_link_libraries(common INTERFACE UCP::libucp)
      target_compile_definitions(common INTERFACE GTBENCH_USE_GHEX_UCP)
      if (GHEX_USE_PMIX)
        find_package(PMIx REQUIRED)
        target_link_libraries(common INTERFACE PMIx::libpmix)
        target_compile_definitions(common INTERFACE GTBENCH_USE_GHEX_PMIX)
      endif()
  endif()
  target_compile_definitions(common INTERFACE GTBENCH_USE_GHEX)
elseif(GTBENCH_COMMUNICATION_BACKEND STREQUAL "simple_mpi")
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_link_libraries(common INTERFACE MPI::MPI_CXX)
  target_compile_definitions(common INTERFACE GTBENCH_USE_SIMPLE_MPI)
endif()
if(GTBENCH_BACKEND STREQUAL "cuda" AND GRIDTOOLS_CUDA_COMPILATION_TYPE STREQUAL "NVCC-CUDA")
  target_compile_options(common INTERFACE -arch=${GTBENCH_CUDA_ARCH})
endif()

target_compile_definitions(common INTERFACE
  GTBENCH_BACKEND=${GTBENCH_BACKEND}
  GTBENCH_FLOAT=${GTBENCH_FLOAT}
  GTBENCH_COMMUNICATION_BACKEND=${GTBENCH_COMMUNICATION_BACKEND}
  )
target_link_libraries(common INTERFACE GridTools::gridtools)

# Subdirectories
add_subdirectory(communication)
add_subdirectory(numerics)

# Current directory
compile_as_cuda(convergence_tests.cpp benchmark.cpp)

add_executable(convergence_tests convergence_tests.cpp)
target_link_libraries(convergence_tests advection diffusion communication)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark advection diffusion communication)
