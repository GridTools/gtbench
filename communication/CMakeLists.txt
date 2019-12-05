compile_as_cuda(ghex_comm.cpp simple_mpi.cpp single_node.cpp)

add_library(communication ${GTBENCH_COMMUNICATION_BACKEND}.cpp)
target_link_libraries(communication common GridTools::gridtools)

if(NOT GTBENCH_COMMUNICATION_BACKEND STREQUAL "single_node")
  target_link_libraries(communication MPI::MPI_CXX)
endif()
