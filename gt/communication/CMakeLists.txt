compile_as_cuda(ghex_comm.cpp)

add_library(communication ${GTBENCH_COMMUNICATION_BACKEND}.cpp)
target_link_libraries(communication GridTools::gridtools)

if (NOT GTBENCH_COMMUNICATION_BACKEND STREQUAL "single_node")
    target_link_libraries(communication MPI::MPI_CXX)
endif()