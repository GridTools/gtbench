cmake_minimum_required(VERSION 3.18.1)

file(STRINGS "version.txt" _gtbench_version)
project(GTBench VERSION ${_gtbench_version} LANGUAGES CXX)
unset(_gtbench_version)

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
set(GTBENCH_BPARAMS_VDIFF "" CACHE STRING "Parameters for GridTools backend for vertical diffusion")
set(GTBENCH_BPARAMS_HADV "" CACHE STRING "Parameters for GridTools backend for horizontal advection")
set(GTBENCH_BPARAMS_VADV "" CACHE STRING "Parameters for GridTools backend for vertical advection")

option(GTBENCH_PYTHON_BINDINGS "Build Python bindings" OFF)

set(_gtbench_cuda_enabled OFF)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND GTBENCH_BACKEND STREQUAL "gpu")
  set(GT_CLANG_CUDA_MODE "Clang-CUDA" CACHE STRING "Clang-CUDA or NVCC-CUDA")
  set_property(CACHE GT_CLANG_CUDA_MODE PROPERTY STRINGS "Clang-CUDA" "NVCC-CUDA")
else()
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    set(_gtbench_cuda_enabled ON)
  endif()
endif()

# Find or fetch GridTools
if(NOT _gridtools_already_fetched)
    find_package(GridTools "2" QUIET)
endif()
if(NOT GridTools_FOUND)
  set(_gridtools_repository "https://github.com/GridTools/gridtools.git")
  set(_gridtools_tag        "release_v2.3")
  if(NOT _gridtools_already_fetched)
    message(STATUS "Fetching GridTools ${_gridtools_tag} from ${_gridtools_repository}")
  endif()
  include(FetchContent)
  FetchContent_Declare(
    gridtools
    GIT_REPOSITORY ${_gridtools_repository}
    GIT_TAG        ${_gridtools_tag}
    OVERRIDE_FIND_PACKAGE
  )
  FetchContent_MakeAvailable(gridtools)
  set(_gridtools_already_fetched ON CACHE INTERNAL "")
endif()

# Helper functions
function(compile_as_cuda)
  if(_gtbench_cuda_enabled)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CUDA COMPILE_OPTIONS "--default-stream=per-thread")
  endif()
endfunction()

add_library(gtbench_common INTERFACE)
target_include_directories(gtbench_common INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:include>
)
target_compile_definitions(gtbench_common INTERFACE
  GTBENCH_BACKEND=${GTBENCH_BACKEND}
  GTBENCH_BACKEND_$<UPPER_CASE:${GTBENCH_BACKEND}>
  GTBENCH_FLOAT=${GTBENCH_FLOAT}
  )
target_link_libraries(gtbench_common INTERFACE
  GridTools::gridtools
  GridTools::stencil_${GTBENCH_BACKEND}
  GridTools::storage_${GTBENCH_BACKEND}
)
set_target_properties(
  gtbench_common
  PROPERTIES
    INTERFACE_POSITION_INDEPENDENT_CODE ON)

add_library(gtbench)
add_library(GTBench::gtbench ALIAS gtbench)
target_link_libraries(gtbench PUBLIC gtbench_common)

add_subdirectory(src)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/export.cmake)

if(GTBENCH_PYTHON_BINDINGS)
  add_subdirectory(python)
endif()
