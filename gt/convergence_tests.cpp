#include "./execution/convergence_tests.hpp"
#include "./communication/backends.hpp"

int main(int argc, char **argv) {
  execution::run_convergence_tests(
      communication::GTBENCH_COMMUNICATION_BACKEND::world(argc, argv));

  return 0;
}