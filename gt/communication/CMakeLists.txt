add_library(communication_single_node single_node.cpp)
target_link_libraries(communication_single_node GridTools::gridtools)

add_library(communication_simple_mpi simple_mpi.cpp)
target_link_libraries(communication_simple_mpi GridTools::gridtools MPI::MPI_CXX)

if (USE_GHEX)
    compile_as_cuda(ghex_comm.cpp)
    add_library(communication_ghex ghex_comm.cpp)
    target_link_libraries(communication_ghex GridTools::gridtools MPI::MPI_CXX)
endif()
