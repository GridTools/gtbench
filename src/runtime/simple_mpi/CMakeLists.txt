find_package(MPI REQUIRED COMPONENTS CXX)

compile_as_cuda(run.cpp)
target_sources(gtbench PRIVATE run.cpp)
target_link_libraries(gtbench PUBLIC MPI::MPI_CXX)
