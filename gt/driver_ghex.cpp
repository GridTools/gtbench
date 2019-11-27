#include "communication/ghex_comm.hpp"
#include "numerics/convergence_tests.hpp"

int main(int argc, char **argv) {
  run_convergence_tests(communication::ghex_comm::world(argc, argv));

  return 0;
}
