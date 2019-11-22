#include "communication/single_node.hpp"
#include "numerics/convergence_tests.hpp"

int main(int argc, char **argv) {
  auto comm_world =
      communication::world(communication::single_node::tag{}, argc, argv);

  run_convergence_tests(comm_world);

  return 0;
}