#include "./communication/single_node.hpp"
#include "./numerics/convergence_tests.hpp"

int main(int argc, char **argv) {
  run_convergence_tests(communication::single_node::world{});

  return 0;
}