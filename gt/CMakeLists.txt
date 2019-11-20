cmake_minimum_required(VERSION 3.14.5)
project(benchmark LANGUAGES CXX)

find_package(GridTools 1.1.0)
find_package(MPI)

add_library(diffusion diffusion.cpp)
target_link_libraries(diffusion GridTools::gridtools)


add_library(advection advection.cpp)
target_link_libraries(advection GridTools::gridtools)

add_executable(driver driver.cpp)
target_link_libraries(driver advection diffusion GridTools::gridtools)

add_executable(mpi_driver mpi_driver.cpp)
target_link_libraries(mpi_driver advection diffusion GridTools::gridtools MPI::MPI_CXX)
