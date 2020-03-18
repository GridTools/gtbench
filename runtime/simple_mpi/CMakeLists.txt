compile_as_cuda(run.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

add_library(runtime run.cpp)
target_link_libraries(runtime common MPI::MPI_CXX)