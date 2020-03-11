compile_as_cuda(ghex_comm.cpp simple_mpi.cpp single_node.cpp gcl.cpp device.cpp)

add_library(communication ${GTBENCH_COMMUNICATION_BACKEND}.cpp)
target_link_libraries(communication common)

add_library(device device.cpp)
target_link_libraries(device common)
