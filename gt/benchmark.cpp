#include <iostream>

#include "./communication/backends.hpp"
#include "./execution/run.hpp"
#include "./numerics/solver.hpp"

int main(int argc, char **argv) {
  auto comm_world =
      communication::GTBENCH_COMMUNICATION_BACKEND::world(argc, argv);

  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " N" << std::endl;
    return 1;
  }

  const std::size_t n = std::atoll(argv[1]);
  const std::size_t nz = 100;

  std::cout << "running benchmark with " << n << "x" << n << " columns"
            << std::endl;

  auto comm_grid = communication::grid(comm_world, {n, n, nz});

  real_t diffusion_coeff = 0.05;
  auto exact = verification::analytical::repeat(
      verification::analytical::advection_diffusion{diffusion_coeff},
      {(n + nz - 1) / nz, (n + nz - 1) / nz, 1});

  auto result = execution::run(
      comm_grid, numerics::advdiff_stepper(diffusion_coeff), 1, 1e-3, exact);
  std::cout << "error: " << result.error << std::endl
            << "time: " << result.time << "s" << std::endl;

  return 0;
}
