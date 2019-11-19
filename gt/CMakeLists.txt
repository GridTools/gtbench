cmake_minimum_required(VERSION 3.14.5)
project(benchmark LANGUAGES CXX)

find_package(GridTools 1.1.0)

add_library(diffusion diffusion.cpp)
target_link_libraries(diffusion GridTools::gridtools)


add_library(advection advection.cpp)
target_link_libraries(advection GridTools::gridtools)

add_executable(driver driver.cpp)
target_link_libraries(driver advection diffusion GridTools::gridtools)
