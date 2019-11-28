#include <iostream>

#include "./communication/simple_mpi.hpp"
#include "./numerics/solver.hpp"
#include "./verification/run.hpp"

int main(int argc, char **argv) {
  auto comm_world = communication::simple_mpi::world(argc, argv);

  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " N" << std::endl;
    return 1;
  }

  const std::size_t n = std::atoll(argv[1]);
  const std::size_t nz = 100;

  std::cout << "running benchmark with " << n << "x" << n << " columns"
            << std::endl;

  auto comm_grid = communication::grid(comm_world, {n, n, 100});

  real_t diffusion_coeff = 0.05;
  auto exact =
      analytical::repeat(analytical::advection_diffusion{diffusion_coeff},
                         {(n + nz - 1) / nz, (n + nz - 1) / nz, 1});

  auto result =
      run(std::move(comm_grid), full_stepper(diffusion_coeff), 1, 1e-3, exact);
  std::cout << "error: " << result.error << std::endl
            << "time: " << result.time << "s" << std::endl;

  return 0;
}