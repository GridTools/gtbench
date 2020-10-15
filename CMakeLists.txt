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

set(GTBENCH_BPARAMS_HDIFF "" CACHE STRING "Parameters for GridTools backend for horizontal diffusion")
set(GTBENCH_BPARAMS_VDIFF1 "" CACHE STRING "Parameters for GridTools backend for vertical diffusion (computation 1)")
set(GTBENCH_BPARAMS_VDIFF2 "" CACHE STRING "Parameters for GridTools backend for vertical diffusion (computation 2)")
set(GTBENCH_BPARAMS_RKADV1 "" CACHE STRING "Parameters for GridTools backend for Runge-Kutta advection step (computation 1)")
set(GTBENCH_BPARAMS_RKADV2 "" CACHE STRING "Parameters for GridTools backend for Runge-Kutta advection step (computation 2)")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND GTBENCH_BACKEND STREQUAL "gpu")
  set(GT_CLANG_CUDA_MODE "Clang-CUDA" CACHE STRING "Clang-CUDA or NVCC-CUDA")
  set_property(CACHE GT_CLANG_CUDA_MODE PROPERTY STRINGS "Clang-CUDA" "NVCC-CUDA")
else()
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_EXTENSIONS OFF)
  endif()
endif()

# Find or fetch GridTools
set(_required_gridtools_version "2")
if(NOT _gridtools_already_fetched)
    find_package(GridTools ${_required_gridtools_version} QUIET)
endif()
if(NOT GridTools_FOUND)
    set(_gridtools_repository "https://github.com/GridTools/gridtools.git")
    set(_gridtools_tag        "v${_required_gridtools_version}.0.0")
    if(NOT _gridtools_already_fetched)
        message(STATUS "Fetching GridTools ${_gridtools_tag} from ${_gridtools_repository}")
    endif()
    include(FetchContent)
    FetchContent_Declare(
        gridtools
        GIT_REPOSITORY ${_gridtools_repository}
        GIT_TAG        ${_gridtools_tag}
    )
    FetchContent_MakeAvailable(gridtools)
    set(_gridtools_already_fetched ON CACHE INTERNAL "")
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
  GTBENCH_BACKEND_$<UPPER_CASE:${GTBENCH_BACKEND}>
  GTBENCH_FLOAT=${GTBENCH_FLOAT}
  )
target_link_libraries(common INTERFACE
  GridTools::gridtools
  GridTools::stencil_${GTBENCH_BACKEND}
  GridTools::storage_${GTBENCH_BACKEND}
  )

add_library(common_runtime INTERFACE)
target_compile_definitions(common_runtime INTERFACE
  GTBENCH_RUNTIME=${GTBENCH_RUNTIME}
  )
target_link_libraries(common_runtime INTERFACE
  common
  )

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
