#include <iostream>

#include "./communication/backends.hpp"
#include "./communication/ghex_comm_mt.hpp"
#include "./execution/run.hpp"
#include "./numerics/solver.hpp"

#include <thread>

int main(int argc, char **argv) {
  //auto comm_world = communication::GTBENCH_COMMUNICATION_BACKEND::world(argc, argv);
  auto comm_world = communication::ghex_comm::world(argc, argv);

  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " N" << std::endl;
    return 1;
  }

  const std::size_t n = std::atoll(argv[1]);
  const std::size_t nz = 100;

  std::cout << "running benchmark with " << n << "x" << n << " columns"
            << std::endl;

  int num_threads = 4;

  //auto comm_grid = communication::grid(comm_world, {n, n, 100});
  auto comm_grid = communication::ghex_comm::comm_grid(comm_world, {n, n, 100});
  auto cc = communication::ghex_comm::grid_mt({n,n,100}, num_threads);
  
  real_t diffusion_coeff = 0.05;
  auto exact = verification::analytical::repeat(
      verification::analytical::advection_diffusion{diffusion_coeff},
      {(n + nz - 1) / nz, (n + nz - 1) / nz, 1});

  {
  std::vector<communication::ghex_comm::grid_mt::sub_grid> sub_grids;
  sub_grids.reserve(num_threads);

  for (int i=0; i<num_threads; ++i)
  {
    sub_grids.push_back( cc[i] );
    std::cout << "offset     = " << sub_grids.back().offset.x << ", " << sub_grids.back().offset.y << std::endl;
    std::cout << "resolution = " << sub_grids.back().resolution.x << ", " << sub_grids.back().resolution.y << ", " << sub_grids.back().resolution.z << std::endl;
    std::cout << "global     = " << sub_grids.back().global_resolution.x << ", " << sub_grids.back().global_resolution.y << std::endl;
  }

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  std::vector<double> errors(num_threads);


  auto execution_func = [diffusion_coeff, &exact](communication::ghex_comm::grid_mt::sub_grid& g, double& err)
  {
    const auto result = execution::run_( g, numerics::advdiff_stepper(diffusion_coeff), 0.1, 1e-3, exact);
    err = result.error;
  };


  for (int i=0; i<num_threads; ++i)
  {
    threads.push_back( std::thread{ execution_func, std::ref(sub_grids[i]), std::ref(errors[i]) } );
  }

  for (auto& t : threads)
    t.join();

  double error = errors[0];
  for (int i=1; i<num_threads; ++i)
      error = std::max(errors[i],error);
    std::cout << "error = " << error << std::endl;
  }


  auto result = execution::run(
      comm_grid, numerics::advdiff_stepper(diffusion_coeff), 0.1, 1e-3, exact);
  std::cout << "error: " << result.error << std::endl
            << "time: " << result.time << "s" << std::endl;

  return 0;
}
