cmake_minimum_required(VERSION 3.14.5)
project(benchmark LANGUAGES CXX)

find_package(GridTools 1.1.0)

add_executable(driver driver.cpp)
target_link_libraries(driver GridTools::gridtools)
