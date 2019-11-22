#include "communication/simple_mpi.hpp"
#include "numerics/convergence_tests.hpp"

int main(int argc, char **argv) {
  auto comm_world =
      communication::world(communication::simple_mpi::tag{}, argc, argv);

  run_convergence_tests(comm_world);

  return 0;
}