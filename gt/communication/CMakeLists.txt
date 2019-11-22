add_library(communication_single_node single_node.cpp)
target_link_libraries(communication_single_node GridTools::gridtools)

add_library(communication_mpi mpi_comm.cpp)
target_link_libraries(communication_mpi GridTools::gridtools MPI::MPI_CXX)