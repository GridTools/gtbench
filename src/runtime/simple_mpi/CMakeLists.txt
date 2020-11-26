find_package(MPI REQUIRED COMPONENTS CXX)

compile_as_cuda(run.cpp TARGET_DIRECTORY runtime)
target_sources(runtime PRIVATE run.cpp)
target_link_libraries(runtime PUBLIC MPI::MPI_CXX)
