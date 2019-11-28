cmake_minimum_required(VERSION 3.14.5)
project(benchmark LANGUAGES CXX)

find_package(GridTools 1.1.0)
find_package(MPI)

set(GHEX_INCLUDE_DIR "" CACHE PATH "path to GHEX include directory")
include_directories(${GHEX_INCLUDE_DIR})

set(GTBENCH_BACKEND "x86" CACHE STRING "Benchmark backend")
set_property(CACHE GTBENCH_BACKEND PROPERTY STRINGS "x86" "mc" "cuda")
set(GTBENCH_FLOAT "double" CACHE STRING "Floating-point type")
set_property(CACHE GTBENCH_FLOAT PROPERTY STRINGS "float" "double")

add_compile_definitions(GTBENCH_BACKEND=${GTBENCH_BACKEND}
                        GTBENCH_FLOAT=${GTBENCH_FLOAT})

if(GTBENCH_BACKEND STREQUAL "cuda")
    enable_language(CUDA)
endif()

add_subdirectory(communication)
add_subdirectory(numerics)

add_executable(driver_single_node driver_single_node.cpp)
target_link_libraries(driver_single_node advection diffusion communication_single_node)

add_executable(driver_simple_mpi driver_simple_mpi.cpp)
target_link_libraries(driver_simple_mpi advection diffusion communication_simple_mpi)

add_executable(driver_ghex driver_ghex.cpp)
target_link_libraries(driver_ghex advection diffusion communication_ghex)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark advection diffusion communication_simple_mpi)

