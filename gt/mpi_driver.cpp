#include "communication/mpi_comm.hpp"
#include "numerics/convergence_tests.hpp"

int main(int argc, char **argv) {
  auto comm_world = communication::world(communication::mpi::tag{}, argc, argv);

  run_convergence_tests(comm_world);

  return 0;
}