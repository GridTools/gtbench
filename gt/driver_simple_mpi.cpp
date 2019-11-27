#include "./communication/simple_mpi.hpp"
#include "./numerics/convergence_tests.hpp"

int main(int argc, char **argv) {
  run_convergence_tests(communication::simple_mpi::world(argc, argv));

  return 0;
}