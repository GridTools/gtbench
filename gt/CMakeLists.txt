cmake_minimum_required(VERSION 3.14.5)
project(benchmark LANGUAGES CXX)

find_package(GridTools 1.1.0)
find_package(MPI)

set(GHEX_INCLUDE_DIR "" CACHE PATH "path to GHEX include directory")
include_directories(${GHEX_INCLUDE_DIR})

add_subdirectory(communication)
add_subdirectory(numerics)

add_executable(driver_single_node driver_single_node.cpp)
target_link_libraries(driver_single_node advection diffusion communication_single_node)

add_executable(driver_simple_mpi driver_simple_mpi.cpp)
target_link_libraries(driver_simple_mpi advection diffusion communication_simple_mpi)

add_executable(driver_ghex driver_ghex.cpp)
target_link_libraries(driver_ghex advection diffusion communication_ghex)
