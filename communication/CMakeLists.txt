compile_as_cuda(ghex_comm.cpp simple_mpi.cpp single_node.cpp)

add_library(communication ${GTBENCH_COMMUNICATION_BACKEND}.cpp)
target_link_libraries(communication common)

if(GTBENCH_COMMUNICATION_BACKEND STREQUAL "ghex_comm")
  add_library(communication_mt ghex_comm_mt.cpp)
  target_link_libraries(communication_mt common)
endif()
