compile_as_cuda(ghex_comm.cpp simple_mpi.cpp single_node.cpp gcl.cpp)

add_library(communication ${GTBENCH_COMMUNICATION_BACKEND}.cpp)
target_link_libraries(communication common)
